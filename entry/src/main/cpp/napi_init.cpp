#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "napi/native_api.h"
#include "ncnn/cpu.h"
#include "ncnn/gpu.h"
#include "ncnn/net.h"
#include "ncnn/platform.h"

namespace {

struct Prediction {
    int index;
    float score;
};

struct OcrDecodeResult {
    std::string text;
    float confidence;
};

struct TextBox {
    int x0;
    int y0;
    int x1;
    int y1;
    float score;
};

napi_value MakeString(napi_env env, const std::string& value)
{
    napi_value result = nullptr;
    napi_create_string_utf8(env, value.c_str(), value.size(), &result);
    return result;
}

bool GetArrayBuffer(napi_env env, napi_value value, const void** data, size_t* size)
{
    bool is_array_buffer = false;
    napi_is_arraybuffer(env, value, &is_array_buffer);
    if (is_array_buffer) {
        void* raw = nullptr;
        napi_get_arraybuffer_info(env, value, &raw, size);
        *data = raw;
        return raw != nullptr && *size > 0;
    }

    bool is_typed_array = false;
    napi_is_typedarray(env, value, &is_typed_array);
    if (!is_typed_array) {
        return false;
    }

    napi_typedarray_type type;
    size_t length = 0;
    void* raw = nullptr;
    napi_value array_buffer = nullptr;
    size_t offset = 0;
    napi_get_typedarray_info(env, value, &type, &length, &raw, &array_buffer, &offset);
    *data = raw;
    *size = length;
    return raw != nullptr && length > 0;
}

bool GetInt32(napi_env env, napi_value value, int32_t* number)
{
    napi_valuetype type;
    napi_typeof(env, value, &type);
    if (type != napi_number) {
        return false;
    }
    return napi_get_value_int32(env, value, number) == napi_ok;
}

bool GetBool(napi_env env, napi_value value, bool* flag)
{
    napi_valuetype type;
    napi_typeof(env, value, &type);
    if (type != napi_boolean) {
        return false;
    }
    return napi_get_value_bool(env, value, flag) == napi_ok;
}

std::vector<float> BuildDemoInput()
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 1;
    std::vector<float> input(width * height * channels, 0.0f);
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const float center_x = static_cast<float>(x - 32);
                const float center_y = static_cast<float>(y - 32);
                const float stroke_a = std::exp(-(center_x * center_x) / 180.0f);
                const float stroke_b = std::exp(-((center_y - center_x * 0.35f) * (center_y - center_x * 0.35f)) / 42.0f);
                input[c * width * height + y * width + x] = std::min(1.0f, stroke_a * 0.65f + stroke_b * 0.55f);
            }
        }
    }
    return input;
}

float RgbaGray(const unsigned char* rgba, int pixel_index)
{
    const int offset = pixel_index * 4;
    const float r = static_cast<float>(rgba[offset]);
    const float g = static_cast<float>(rgba[offset + 1]);
    const float b = static_cast<float>(rgba[offset + 2]);
    return r * 0.299f + g * 0.587f + b * 0.114f;
}

std::vector<float> BuildInputFromRgba(const unsigned char* rgba, int width, int height)
{
    constexpr int target_width = 64;
    constexpr int target_height = 64;
    std::vector<float> input(target_width * target_height, 0.0f);
    if (rgba == nullptr || width <= 0 || height <= 0) {
        return BuildDemoInput();
    }

    int min_x = width;
    int min_y = height;
    int max_x = -1;
    int max_y = -1;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float gray = RgbaGray(rgba, y * width + x);
            if (gray < 235.0f) {
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
            }
        }
    }

    if (max_x < min_x || max_y < min_y) {
        min_x = 0;
        min_y = 0;
        max_x = width - 1;
        max_y = height - 1;
    } else {
        const int pad_x = std::max(2, (max_x - min_x + 1) / 12);
        const int pad_y = std::max(2, (max_y - min_y + 1) / 12);
        min_x = std::max(0, min_x - pad_x);
        min_y = std::max(0, min_y - pad_y);
        max_x = std::min(width - 1, max_x + pad_x);
        max_y = std::min(height - 1, max_y + pad_y);
    }

    const int crop_w = std::max(1, max_x - min_x + 1);
    const int crop_h = std::max(1, max_y - min_y + 1);
    for (int ty = 0; ty < target_height; ++ty) {
        const float sy = static_cast<float>(min_y) + (static_cast<float>(ty) + 0.5f) * crop_h / target_height - 0.5f;
        const int y0 = std::max(0, std::min(height - 1, static_cast<int>(std::floor(sy))));
        const int y1 = std::max(0, std::min(height - 1, y0 + 1));
        const float fy = sy - std::floor(sy);
        for (int tx = 0; tx < target_width; ++tx) {
            const float sx = static_cast<float>(min_x) + (static_cast<float>(tx) + 0.5f) * crop_w / target_width - 0.5f;
            const int x0 = std::max(0, std::min(width - 1, static_cast<int>(std::floor(sx))));
            const int x1 = std::max(0, std::min(width - 1, x0 + 1));
            const float fx = sx - std::floor(sx);

            const float g00 = RgbaGray(rgba, y0 * width + x0);
            const float g01 = RgbaGray(rgba, y0 * width + x1);
            const float g10 = RgbaGray(rgba, y1 * width + x0);
            const float g11 = RgbaGray(rgba, y1 * width + x1);
            const float top = g00 * (1.0f - fx) + g01 * fx;
            const float bottom = g10 * (1.0f - fx) + g11 * fx;
            const float gray = top * (1.0f - fy) + bottom * fy;
            input[ty * target_width + tx] = std::max(0.0f, std::min(1.0f, (255.0f - gray) / 255.0f));
        }
    }
    return input;
}

ncnn::Mat BuildPpocrInputFromRgba(const unsigned char* rgba, int width, int height)
{
    constexpr int target_height = 48;
    constexpr int min_width = 128;
    constexpr int max_width = 1024;
    if (rgba == nullptr || width <= 0 || height <= 0) {
        return ncnn::Mat();
    }

    int min_x = width;
    int min_y = height;
    int max_x = -1;
    int max_y = -1;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float gray = RgbaGray(rgba, y * width + x);
            if (gray < 245.0f) {
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
            }
        }
    }

    if (max_x < min_x || max_y < min_y) {
        min_x = 0;
        min_y = 0;
        max_x = width - 1;
        max_y = height - 1;
    } else {
        const int pad_x = std::max(4, (max_x - min_x + 1) / 10);
        const int pad_y = std::max(4, (max_y - min_y + 1) / 10);
        min_x = std::max(0, min_x - pad_x);
        min_y = std::max(0, min_y - pad_y);
        max_x = std::min(width - 1, max_x + pad_x);
        max_y = std::min(height - 1, max_y + pad_y);
    }

    const int crop_w = std::max(1, max_x - min_x + 1);
    const int crop_h = std::max(1, max_y - min_y + 1);
    int resized_width = static_cast<int>(std::ceil(target_height * (static_cast<float>(crop_w) / crop_h)));
    resized_width = std::max(1, std::min(max_width, resized_width));
    int input_width = std::max(min_width, resized_width);
    input_width = std::min(max_width, input_width);

    ncnn::Mat input(input_width, target_height, 3);
    for (int c = 0; c < 3; ++c) {
        float* channel = input.channel(c);
        std::fill(channel, channel + input_width * target_height, 0.0f);
    }

    for (int ty = 0; ty < target_height; ++ty) {
        const float sy = static_cast<float>(min_y) + (static_cast<float>(ty) + 0.5f) * crop_h / target_height - 0.5f;
        const int y0 = std::max(0, std::min(height - 1, static_cast<int>(std::floor(sy))));
        const int y1 = std::max(0, std::min(height - 1, y0 + 1));
        const float fy = sy - std::floor(sy);
        for (int tx = 0; tx < resized_width; ++tx) {
            const float sx = static_cast<float>(min_x) + (static_cast<float>(tx) + 0.5f) * crop_w / resized_width - 0.5f;
            const int x0 = std::max(0, std::min(width - 1, static_cast<int>(std::floor(sx))));
            const int x1 = std::max(0, std::min(width - 1, x0 + 1));
            const float fx = sx - std::floor(sx);

            const int offsets[4] = {(y0 * width + x0) * 4, (y0 * width + x1) * 4,
                                    (y1 * width + x0) * 4, (y1 * width + x1) * 4};
            for (int c = 0; c < 3; ++c) {
                const int rgba_channel = c == 0 ? 2 : (c == 1 ? 1 : 0); // PaddleOCR uses OpenCV BGR order.
                const float v00 = rgba[offsets[0] + rgba_channel];
                const float v01 = rgba[offsets[1] + rgba_channel];
                const float v10 = rgba[offsets[2] + rgba_channel];
                const float v11 = rgba[offsets[3] + rgba_channel];
                const float top = v00 * (1.0f - fx) + v01 * fx;
                const float bottom = v10 * (1.0f - fx) + v11 * fx;
                const float value = ((top * (1.0f - fy) + bottom * fy) / 255.0f - 0.5f) / 0.5f;
                input.channel(c)[ty * input_width + tx] = value;
            }
        }
    }

    return input;
}

ncnn::Mat BuildPpocrDetInputFromRgba(const unsigned char* rgba, int width, int height)
{
    constexpr int target_size = 640;
    if (rgba == nullptr || width <= 0 || height <= 0) {
        return ncnn::Mat();
    }

    ncnn::Mat input(target_size, target_size, 3);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};
    for (int ty = 0; ty < target_size; ++ty) {
        const float sy = (static_cast<float>(ty) + 0.5f) * height / target_size - 0.5f;
        const int y0 = std::max(0, std::min(height - 1, static_cast<int>(std::floor(sy))));
        const int y1 = std::max(0, std::min(height - 1, y0 + 1));
        const float fy = sy - std::floor(sy);
        for (int tx = 0; tx < target_size; ++tx) {
            const float sx = (static_cast<float>(tx) + 0.5f) * width / target_size - 0.5f;
            const int x0 = std::max(0, std::min(width - 1, static_cast<int>(std::floor(sx))));
            const int x1 = std::max(0, std::min(width - 1, x0 + 1));
            const float fx = sx - std::floor(sx);
            const int offsets[4] = {(y0 * width + x0) * 4, (y0 * width + x1) * 4,
                                    (y1 * width + x0) * 4, (y1 * width + x1) * 4};
            for (int c = 0; c < 3; ++c) {
                const int rgba_channel = c == 0 ? 2 : (c == 1 ? 1 : 0);
                const float v00 = rgba[offsets[0] + rgba_channel];
                const float v01 = rgba[offsets[1] + rgba_channel];
                const float v10 = rgba[offsets[2] + rgba_channel];
                const float v11 = rgba[offsets[3] + rgba_channel];
                const float top = v00 * (1.0f - fx) + v01 * fx;
                const float bottom = v10 * (1.0f - fx) + v11 * fx;
                input.channel(c)[ty * target_size + tx] = ((top * (1.0f - fy) + bottom * fy) / 255.0f - mean[c]) / stdv[c];
            }
        }
    }
    return input;
}

ncnn::Mat BuildPpocrRecInputFromBox(const unsigned char* rgba, int width, int height, const TextBox& box)
{
    constexpr int target_height = 48;
    constexpr int min_width = 128;
    constexpr int max_width = 1024;
    const int pad_x = std::max(4, (box.x1 - box.x0 + 1) / 12);
    const int pad_y = std::max(4, (box.y1 - box.y0 + 1) / 8);
    const int min_x = std::max(0, box.x0 - pad_x);
    const int min_y = std::max(0, box.y0 - pad_y);
    const int max_x = std::min(width - 1, box.x1 + pad_x);
    const int max_y = std::min(height - 1, box.y1 + pad_y);
    const int crop_w = std::max(1, max_x - min_x + 1);
    const int crop_h = std::max(1, max_y - min_y + 1);
    int resized_width = static_cast<int>(std::ceil(target_height * (static_cast<float>(crop_w) / crop_h)));
    resized_width = std::max(1, std::min(max_width, resized_width));
    int input_width = std::max(min_width, resized_width);
    input_width = std::min(max_width, input_width);

    ncnn::Mat input(input_width, target_height, 3);
    for (int c = 0; c < 3; ++c) {
        float* channel = input.channel(c);
        std::fill(channel, channel + input_width * target_height, 0.0f);
    }

    for (int ty = 0; ty < target_height; ++ty) {
        const float sy = static_cast<float>(min_y) + (static_cast<float>(ty) + 0.5f) * crop_h / target_height - 0.5f;
        const int y0 = std::max(0, std::min(height - 1, static_cast<int>(std::floor(sy))));
        const int y1 = std::max(0, std::min(height - 1, y0 + 1));
        const float fy = sy - std::floor(sy);
        for (int tx = 0; tx < resized_width; ++tx) {
            const float sx = static_cast<float>(min_x) + (static_cast<float>(tx) + 0.5f) * crop_w / resized_width - 0.5f;
            const int x0 = std::max(0, std::min(width - 1, static_cast<int>(std::floor(sx))));
            const int x1 = std::max(0, std::min(width - 1, x0 + 1));
            const float fx = sx - std::floor(sx);
            const int offsets[4] = {(y0 * width + x0) * 4, (y0 * width + x1) * 4,
                                    (y1 * width + x0) * 4, (y1 * width + x1) * 4};
            for (int c = 0; c < 3; ++c) {
                const int rgba_channel = c == 0 ? 2 : (c == 1 ? 1 : 0);
                const float v00 = rgba[offsets[0] + rgba_channel];
                const float v01 = rgba[offsets[1] + rgba_channel];
                const float v10 = rgba[offsets[2] + rgba_channel];
                const float v11 = rgba[offsets[3] + rgba_channel];
                const float top = v00 * (1.0f - fx) + v01 * fx;
                const float bottom = v10 * (1.0f - fx) + v11 * fx;
                input.channel(c)[ty * input_width + tx] = ((top * (1.0f - fy) + bottom * fy) / 255.0f - 0.5f) / 0.5f;
            }
        }
    }
    return input;
}

std::vector<TextBox> DetectTextBoxes(const ncnn::Mat& map, int image_width, int image_height)
{
    std::vector<TextBox> boxes;
    if (map.empty() || map.w <= 0 || map.h <= 0 || image_width <= 0 || image_height <= 0) {
        return boxes;
    }

    const int w = map.w;
    const int h = map.h;
    const float* scores = map.channel(0);
    constexpr float pixel_thresh = 0.20f;
    constexpr float box_thresh = 0.45f;
    std::vector<unsigned char> visited(static_cast<size_t>(w) * h, 0);
    std::vector<int> stack;
    stack.reserve(4096);

    for (int sy = 0; sy < h; ++sy) {
        for (int sx = 0; sx < w; ++sx) {
            const int start = sy * w + sx;
            if (visited[start] || scores[start] < pixel_thresh) {
                continue;
            }

            int min_x = sx;
            int min_y = sy;
            int max_x = sx;
            int max_y = sy;
            float score_sum = 0.0f;
            int count = 0;
            stack.clear();
            stack.push_back(start);
            visited[start] = 1;
            while (!stack.empty()) {
                const int idx = stack.back();
                stack.pop_back();
                const int x = idx % w;
                const int y = idx / w;
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
                score_sum += scores[idx];
                ++count;

                const int dx[4] = {1, -1, 0, 0};
                const int dy[4] = {0, 0, 1, -1};
                for (int i = 0; i < 4; ++i) {
                    const int nx = x + dx[i];
                    const int ny = y + dy[i];
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) {
                        continue;
                    }
                    const int ni = ny * w + nx;
                    if (!visited[ni] && scores[ni] >= pixel_thresh) {
                        visited[ni] = 1;
                        stack.push_back(ni);
                    }
                }
            }

            const int bw = max_x - min_x + 1;
            const int bh = max_y - min_y + 1;
            const float score = count > 0 ? score_sum / count : 0.0f;
            if (score < box_thresh || bw < 4 || bh < 4 || count < 16) {
                continue;
            }

            const int pad_x = std::max(2, bw / 8);
            const int pad_y = std::max(2, bh / 6);
            min_x = std::max(0, min_x - pad_x);
            min_y = std::max(0, min_y - pad_y);
            max_x = std::min(w - 1, max_x + pad_x);
            max_y = std::min(h - 1, max_y + pad_y);
            TextBox box;
            box.x0 = std::max(0, std::min(image_width - 1, static_cast<int>(std::floor(static_cast<float>(min_x) * image_width / w))));
            box.y0 = std::max(0, std::min(image_height - 1, static_cast<int>(std::floor(static_cast<float>(min_y) * image_height / h))));
            box.x1 = std::max(0, std::min(image_width - 1, static_cast<int>(std::ceil(static_cast<float>(max_x + 1) * image_width / w)) - 1));
            box.y1 = std::max(0, std::min(image_height - 1, static_cast<int>(std::ceil(static_cast<float>(max_y + 1) * image_height / h)) - 1));
            box.score = score;
            if (box.x1 - box.x0 > 3 && box.y1 - box.y0 > 3) {
                boxes.push_back(box);
            }
        }
    }

    std::sort(boxes.begin(), boxes.end(), [](const TextBox& lhs, const TextBox& rhs) {
        const int lh = lhs.y1 - lhs.y0 + 1;
        const int rh = rhs.y1 - rhs.y0 + 1;
        if (std::abs(lhs.y0 - rhs.y0) > std::max(lh, rh) / 2) {
            return lhs.y0 < rhs.y0;
        }
        return lhs.x0 < rhs.x0;
    });
    if (boxes.size() > 32) {
        boxes.resize(32);
    }
    return boxes;
}

std::vector<std::string> ParseLabels(const char* data, size_t size)
{
    std::vector<std::string> labels;
    std::string current;
    labels.reserve(20000);
    for (size_t i = 0; i < size; ++i) {
        const char ch = data[i];
        if (ch == '\n') {
            if (!current.empty() && current.back() == '\r') {
                current.pop_back();
            }
            labels.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        if (current.back() == '\r') {
            current.pop_back();
        }
        labels.push_back(current);
    }
    return labels;
}

OcrDecodeResult DecodeCtc(const ncnn::Mat& output, const std::vector<std::string>& labels)
{
    OcrDecodeResult result;
    if (output.dims != 2 || output.w <= 0 || output.h <= 0) {
        result.confidence = 0.0f;
        return result;
    }

    int last_index = -1;
    float score_sum = 0.0f;
    int score_count = 0;
    for (int t = 0; t < output.h; ++t) {
        const float* row = output.row(t);
        int best_index = 0;
        float best_score = row[0];
        for (int i = 1; i < output.w; ++i) {
            if (row[i] > best_score) {
                best_score = row[i];
                best_index = i;
            }
        }

        if (best_index != 0 && best_index != last_index) {
            if (best_index >= 1 && best_index <= static_cast<int>(labels.size())) {
                result.text += labels[best_index - 1];
                score_sum += best_score;
                ++score_count;
            } else if (best_index == static_cast<int>(labels.size()) + 1) {
                result.text += " ";
                score_sum += best_score;
                ++score_count;
            }
        }
        last_index = best_index;
    }
    result.confidence = score_count > 0 ? score_sum / score_count : 0.0f;
    return result;
}

std::vector<Prediction> TopK(const ncnn::Mat& output, int k)
{
    std::vector<Prediction> predictions;
    predictions.reserve(output.w);
    for (int i = 0; i < output.w; ++i) {
        predictions.push_back({i, output[i]});
    }

    const int count = std::min(k, static_cast<int>(predictions.size()));
    std::partial_sort(predictions.begin(), predictions.begin() + count, predictions.end(),
                      [](const Prediction& lhs, const Prediction& rhs) {
                          return lhs.score > rhs.score;
                      });
    predictions.resize(count);
    return predictions;
}

std::string JsonEscape(const std::string& value)
{
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (unsigned char ch : value) {
        if (ch == '\\' || ch == '"') {
            escaped.push_back('\\');
            escaped.push_back(static_cast<char>(ch));
        } else if (ch == '\n') {
            escaped += "\\n";
        } else if (ch == '\r') {
            escaped += "\\r";
        } else if (ch == '\t') {
            escaped += "\\t";
        } else if (ch < 0x20) {
            char buf[7];
            std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
            escaped += buf;
        } else {
            escaped.push_back(static_cast<char>(ch));
        }
    }
    return escaped;
}

struct LoadNetResult {
    int param_ret;
    int model_ret;
    bool requested_gpu;
    bool effective_gpu;
};

struct GpuRuntimeStatus {
    int instance_ret;
    int gpu_count;
    bool available;
};

GpuRuntimeStatus QueryGpuRuntime()
{
    GpuRuntimeStatus status;
    status.instance_ret = -1;
    status.gpu_count = 0;
    status.available = false;
#if NCNN_VULKAN
    status.instance_ret = ncnn::create_gpu_instance();
    if (status.instance_ret == 0) {
        status.gpu_count = ncnn::get_gpu_count();
        status.available = status.gpu_count > 0;
    }
#endif
    return status;
}

LoadNetResult LoadNcnnNet(ncnn::Net& net, const void* param_data, size_t param_size, const void* model_data, bool request_gpu)
{
    std::vector<unsigned char> param_bytes(static_cast<const unsigned char*>(param_data),
                                           static_cast<const unsigned char*>(param_data) + param_size);
    param_bytes.push_back('\0');
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(1);
    net.opt.num_threads = 1;
    const GpuRuntimeStatus gpu_status = QueryGpuRuntime();
    const bool effective_gpu = request_gpu && gpu_status.available;
    net.opt.use_vulkan_compute = effective_gpu;
    if (effective_gpu) {
        net.set_vulkan_device(0);
    }
    LoadNetResult result;
    result.requested_gpu = request_gpu;
    result.effective_gpu = effective_gpu;
    result.param_ret = net.load_param_mem(reinterpret_cast<const char*>(param_bytes.data()));
    result.model_ret = result.param_ret == 0 ? net.load_model(static_cast<const unsigned char*>(model_data)) : -999;
    return result;
}

std::string LoadErrorJson(const char* model_name, const LoadNetResult& result, size_t param_size, size_t model_size)
{
    std::ostringstream json;
    json << "{\"ok\":false,\"error\":\"failed to load " << model_name
         << "\",\"paramRet\":" << result.param_ret
         << ",\"modelRet\":" << result.model_ret
         << ",\"requestedGpu\":" << (result.requested_gpu ? "true" : "false")
         << ",\"effectiveGpu\":" << (result.effective_gpu ? "true" : "false")
         << ",\"paramBytes\":" << param_size
         << ",\"modelBytes\":" << model_size << "}";
    return json.str();
}

napi_value GetRuntimeInfo(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value argv[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    bool request_gpu = false;
    if (argc >= 1) {
        GetBool(env, argv[0], &request_gpu);
    }
    const GpuRuntimeStatus gpu_status = QueryGpuRuntime();
    const bool effective_gpu = request_gpu && gpu_status.available;
#if NCNN_VULKAN
    const bool vulkan_compiled = true;
#else
    const bool vulkan_compiled = false;
#endif

    std::ostringstream json;
    json << "{\"ok\":true"
         << ",\"ncnnVersion\":\"" << NCNN_VERSION_STRING << "\""
         << ",\"vulkanCompiled\":" << (vulkan_compiled ? "true" : "false")
         << ",\"gpuInstanceRet\":" << gpu_status.instance_ret
         << ",\"gpuCount\":" << gpu_status.gpu_count
         << ",\"gpuRequested\":" << (request_gpu ? "true" : "false")
         << ",\"inferenceBackend\":\"" << (effective_gpu ? "Vulkan GPU" : "CPU") << "\""
         << ",\"useVulkanCompute\":" << (effective_gpu ? "true" : "false")
         << ",\"numThreads\":1"
         << "}";
    return MakeString(env, json.str());
}

napi_value Recognize(napi_env env, napi_callback_info info)
{
    size_t argc = 11;
    napi_value argv[11] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    if (argc < 2) {
        return MakeString(env, "{\"ok\":false,\"error\":\"recognize requires param and model buffers\"}");
    }

    const void* param_data = nullptr;
    const void* model_data = nullptr;
    size_t param_size = 0;
    size_t model_size = 0;
    if (!GetArrayBuffer(env, argv[0], &param_data, &param_size) || !GetArrayBuffer(env, argv[1], &model_data, &model_size)) {
        return MakeString(env, "{\"ok\":false,\"error\":\"invalid ArrayBuffer arguments\"}");
    }

    const void* rgba_data = nullptr;
    size_t rgba_size = 0;
    int32_t image_width = 0;
    int32_t image_height = 0;
    const bool has_image = argc >= 5 &&
                           GetArrayBuffer(env, argv[2], &rgba_data, &rgba_size) &&
                           GetInt32(env, argv[3], &image_width) &&
                           GetInt32(env, argv[4], &image_height) &&
                           rgba_size >= static_cast<size_t>(image_width) * static_cast<size_t>(image_height) * 4;

    const void* labels_data = nullptr;
    size_t labels_size = 0;
    const bool has_labels = argc >= 6 && GetArrayBuffer(env, argv[5], &labels_data, &labels_size);
    bool request_gpu = false;
    if (argc >= 11) {
        GetBool(env, argv[10], &request_gpu);
    }

    const auto started = std::chrono::steady_clock::now();
    ncnn::Mat output;
    std::vector<TextBox> detected_boxes;
    std::vector<OcrDecodeResult> decoded_boxes;
    bool full_ocr = false;
    std::string det_error;

    if (has_labels && argc >= 10) {
        if (!has_image) {
            return MakeString(env, "{\"ok\":false,\"error\":\"PP-OCR requires image pixels\"}");
        }
        const void* det_param_data = nullptr;
        const void* det_model_data = nullptr;
        const void* rec_param_data = nullptr;
        const void* rec_model_data = nullptr;
        size_t det_param_size = 0;
        size_t det_model_size = 0;
        size_t rec_param_size = 0;
        size_t rec_model_size = 0;
        if (!GetArrayBuffer(env, argv[6], &det_param_data, &det_param_size) ||
            !GetArrayBuffer(env, argv[7], &det_model_data, &det_model_size) ||
            !GetArrayBuffer(env, argv[8], &rec_param_data, &rec_param_size) ||
            !GetArrayBuffer(env, argv[9], &rec_model_data, &rec_model_size)) {
            return MakeString(env, "{\"ok\":false,\"error\":\"full OCR requires det and rec model buffers\"}");
        }

        ncnn::Net det_net;
        const LoadNetResult det_load = LoadNcnnNet(det_net, det_param_data, det_param_size, det_model_data, request_gpu);
        if (det_load.param_ret != 0 || det_load.model_ret <= 0) {
            det_error = LoadErrorJson("PP-OCR det model", det_load, det_param_size, det_model_size);
        } else {
            ncnn::Mat det_input = BuildPpocrDetInputFromRgba(static_cast<const unsigned char*>(rgba_data), image_width, image_height);
            if (det_input.empty()) {
                det_error = "{\"ok\":false,\"error\":\"failed to preprocess PP-OCR det input\"}";
            } else {
                ncnn::Extractor det_extractor = det_net.create_extractor();
                ncnn::Mat det_output;
                if (det_extractor.input("in0", det_input) != 0) {
                    det_error = "{\"ok\":false,\"error\":\"failed to feed PP-OCR det input\"}";
                } else if (det_extractor.extract("out0", det_output) != 0) {
                    det_error = "{\"ok\":false,\"error\":\"failed to extract PP-OCR det output\"}";
                } else {
                    detected_boxes = DetectTextBoxes(det_output, image_width, image_height);
                }
            }
        }
        if (detected_boxes.empty()) {
            detected_boxes.push_back({0, 0, image_width - 1, image_height - 1, 0.0f});
        }

        ncnn::Net rec_net;
        const LoadNetResult rec_load = LoadNcnnNet(rec_net, rec_param_data, rec_param_size, rec_model_data, request_gpu);
        if (rec_load.param_ret != 0 || rec_load.model_ret <= 0) {
            return MakeString(env, LoadErrorJson("PP-OCR rec model", rec_load, rec_param_size, rec_model_size));
        }
        const std::vector<std::string> labels = ParseLabels(static_cast<const char*>(labels_data), labels_size);
        for (const TextBox& box : detected_boxes) {
            ncnn::Mat rec_input = BuildPpocrRecInputFromBox(static_cast<const unsigned char*>(rgba_data), image_width, image_height, box);
            if (rec_input.empty()) {
                decoded_boxes.push_back({"", 0.0f});
                continue;
            }
            ncnn::Extractor rec_extractor = rec_net.create_extractor();
            if (rec_extractor.input("in0", rec_input) != 0 || rec_extractor.extract("out0", output) != 0) {
                decoded_boxes.push_back({"", 0.0f});
                continue;
            }
            decoded_boxes.push_back(DecodeCtc(output, labels));
        }
        full_ocr = true;
    } else if (has_labels) {
        if (!has_image) {
            return MakeString(env, "{\"ok\":false,\"error\":\"PP-OCR requires image pixels\"}");
        }
        ncnn::Net net;
        const LoadNetResult load = LoadNcnnNet(net, param_data, param_size, model_data, request_gpu);
        if (load.param_ret != 0 || load.model_ret <= 0) {
            return MakeString(env, LoadErrorJson("PP-OCR rec model", load, param_size, model_size));
        }
        ncnn::Mat input = BuildPpocrInputFromRgba(static_cast<const unsigned char*>(rgba_data), image_width, image_height);
        if (input.empty()) {
            return MakeString(env, "{\"ok\":false,\"error\":\"failed to preprocess PP-OCR input\"}");
        }
        ncnn::Extractor extractor = net.create_extractor();
        if (extractor.input("in0", input) != 0) {
            return MakeString(env, "{\"ok\":false,\"error\":\"failed to feed PP-OCR input\"}");
        }
        if (extractor.extract("out0", output) != 0) {
            return MakeString(env, "{\"ok\":false,\"error\":\"failed to extract PP-OCR output\"}");
        }
    } else {
        ncnn::Net net;
        const LoadNetResult load = LoadNcnnNet(net, param_data, param_size, model_data, request_gpu);
        if (load.param_ret != 0 || load.model_ret <= 0) {
            return MakeString(env, LoadErrorJson("ncnn model", load, param_size, model_size));
        }
        const std::vector<float> input_data = has_image
            ? BuildInputFromRgba(static_cast<const unsigned char*>(rgba_data), image_width, image_height)
            : BuildDemoInput();
        ncnn::Mat input(64, 64, 1);
        std::memcpy(input.channel(0), input_data.data(), input_data.size() * sizeof(float));
        ncnn::Extractor extractor = net.create_extractor();
        if (extractor.input("input", input) != 0) {
            return MakeString(env, "{\"ok\":false,\"error\":\"failed to feed input\"}");
        }
        if (extractor.extract("prob", output) != 0) {
            return MakeString(env, "{\"ok\":false,\"error\":\"failed to extract output\"}");
        }
    }
    const auto finished = std::chrono::steady_clock::now();
    const long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(finished - started).count();

    std::ostringstream json;
    json.setf(std::ios::fixed);
    json.precision(6);
    json << "{\"ok\":true,\"elapsedMs\":" << elapsed_ms;
    if (full_ocr) {
        std::string merged_text;
        float confidence_sum = 0.0f;
        int confidence_count = 0;
        for (const OcrDecodeResult& decoded : decoded_boxes) {
            if (decoded.text.empty()) {
                continue;
            }
            if (!merged_text.empty()) {
                merged_text += "\n";
            }
            merged_text += decoded.text;
            confidence_sum += decoded.confidence;
            ++confidence_count;
        }
        json << ",\"text\":\"" << JsonEscape(merged_text) << "\",\"confidence\":"
             << (confidence_count > 0 ? confidence_sum / confidence_count : 0.0f);
        json << ",\"boxes\":[";
        for (size_t i = 0; i < detected_boxes.size(); ++i) {
            if (i > 0) {
                json << ",";
            }
            const TextBox& box = detected_boxes[i];
            const std::string box_text = i < decoded_boxes.size() ? decoded_boxes[i].text : "";
            const float rec_score = i < decoded_boxes.size() ? decoded_boxes[i].confidence : 0.0f;
            json << "{\"x\":" << box.x0 << ",\"y\":" << box.y0 << ",\"w\":" << (box.x1 - box.x0 + 1)
                 << ",\"h\":" << (box.y1 - box.y0 + 1) << ",\"detScore\":" << box.score
                 << ",\"text\":\"" << JsonEscape(box_text) << "\",\"confidence\":" << rec_score << "}";
        }
        json << "]";
        if (!det_error.empty()) {
            json << ",\"detError\":\"" << JsonEscape(det_error) << "\"";
        }
    } else if (has_labels) {
        const std::vector<std::string> labels = ParseLabels(static_cast<const char*>(labels_data), labels_size);
        const OcrDecodeResult decoded = DecodeCtc(output, labels);
        json << ",\"text\":\"" << JsonEscape(decoded.text) << "\",\"confidence\":" << decoded.confidence;
    } else {
        json << ",\"topK\":[";
        const std::vector<Prediction> top = TopK(output, 5);
        for (size_t i = 0; i < top.size(); ++i) {
            if (i > 0) {
                json << ",";
            }
            json << "{\"index\":" << top[i].index << ",\"score\":" << top[i].score << "}";
        }
        json << "]";
    }
    json << "}";
    return MakeString(env, json.str());
}

napi_value Init(napi_env env, napi_value exports)
{
    napi_property_descriptor desc[] = {
        {"recognize", nullptr, Recognize, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getRuntimeInfo", nullptr, GetRuntimeInfo, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}

} // namespace

EXTERN_C_START
static napi_module hwdb_module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = nullptr,
    .reserved = {0},
};
EXTERN_C_END

extern "C" __attribute__((constructor)) void RegisterHwdbModule()
{
    napi_module_register(&hwdb_module);
}
