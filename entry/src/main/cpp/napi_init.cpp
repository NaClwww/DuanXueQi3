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
#include "ncnn/net.h"

namespace {

struct Prediction {
    int index;
    float score;
};

struct OcrDecodeResult {
    std::string text;
    float confidence;
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
    constexpr int max_width = 512;
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

napi_value Recognize(napi_env env, napi_callback_info info)
{
    size_t argc = 6;
    napi_value argv[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
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

    std::vector<unsigned char> param_bytes(static_cast<const unsigned char*>(param_data),
                                           static_cast<const unsigned char*>(param_data) + param_size);
    param_bytes.push_back('\0');

    ncnn::Net net;
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(1);
    net.opt.num_threads = 1;
    net.opt.use_vulkan_compute = false;

    const auto started = std::chrono::steady_clock::now();
    if (net.load_param_mem(reinterpret_cast<const char*>(param_bytes.data())) != 0) {
        return MakeString(env, "{\"ok\":false,\"error\":\"failed to load ncnn param\"}");
    }
    if (net.load_model(static_cast<const unsigned char*>(model_data)) == 0) {
        return MakeString(env, "{\"ok\":false,\"error\":\"failed to load ncnn model\"}");
    }

    ncnn::Extractor extractor = net.create_extractor();
    ncnn::Mat output;
    if (has_labels) {
        if (!has_image) {
            return MakeString(env, "{\"ok\":false,\"error\":\"PP-OCR requires image pixels\"}");
        }
        ncnn::Mat input = BuildPpocrInputFromRgba(static_cast<const unsigned char*>(rgba_data), image_width, image_height);
        if (input.empty()) {
            return MakeString(env, "{\"ok\":false,\"error\":\"failed to preprocess PP-OCR input\"}");
        }
        if (extractor.input("in0", input) != 0) {
            return MakeString(env, "{\"ok\":false,\"error\":\"failed to feed PP-OCR input\"}");
        }
        if (extractor.extract("out0", output) != 0) {
            return MakeString(env, "{\"ok\":false,\"error\":\"failed to extract PP-OCR output\"}");
        }
    } else {
        const std::vector<float> input_data = has_image
            ? BuildInputFromRgba(static_cast<const unsigned char*>(rgba_data), image_width, image_height)
            : BuildDemoInput();
        ncnn::Mat input(64, 64, 1);
        std::memcpy(input.channel(0), input_data.data(), input_data.size() * sizeof(float));

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
    if (has_labels) {
        const std::vector<std::string> labels = ParseLabels(static_cast<const char*>(labels_data), labels_size);
        const OcrDecodeResult decoded = DecodeCtc(output, labels);
        json << ",\"text\":\"";
        for (char ch : decoded.text) {
            if (ch == '\\' || ch == '"') {
                json << '\\';
            }
            json << ch;
        }
        json << "\",\"confidence\":" << decoded.confidence;
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
