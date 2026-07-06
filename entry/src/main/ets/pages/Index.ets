import abilityAccessCtrl from '@ohos.abilityAccessCtrl';
import type { Permissions } from '@ohos.abilityAccessCtrl';
import camera from '@ohos.multimedia.camera';
import cameraPicker from '@ohos.multimedia.cameraPicker';
import image from '@ohos.multimedia.image';
import nativeRecognizer from 'libentry.so';

interface NativeRecognizer {
  recognize(
    param: ArrayBuffer,
    model: ArrayBuffer,
    rgba?: ArrayBuffer,
    width?: number,
    height?: number,
    labels?: ArrayBuffer
  ): string;
}

interface NativePrediction {
  index: number;
  score: number;
}

interface NativeResult {
  ok: boolean;
  elapsedMs?: number;
  topK?: NativePrediction[];
  text?: string;
  confidence?: number;
  error?: string;
}

interface DecodedImage {
  pixels: ArrayBuffer;
  width: number;
  height: number;
}

const LABELS: string[] = ['一', '二', '三', '人', '大', '小', '中', '国', '木', '水'];

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

@Entry
@Component
struct Index {
  @State status: string = '等待拍照或推理';
  @State elapsedMs: string = '--';
  @State predictions: NativePrediction[] = [];
  @State recognizedText: string = '';
  @State confidence: string = '--';
  @State errorMessage: string = '';
  @State capturedImageUri: string = '';
  @State sampleImageIndex: number = -1;

  private async capturePhotoAndRecognize(): Promise<void> {
    this.status = '正在打开相机';
    this.errorMessage = '';

    try {
      const context = getContext(this);
      const atManager = abilityAccessCtrl.createAtManager();
      await atManager.requestPermissionsFromUser(context, ['ohos.permission.CAMERA'] as Permissions[]);
      const profile: cameraPicker.PickerProfile = {
        cameraPosition: camera.CameraPosition.CAMERA_POSITION_BACK
      };

      const result = await cameraPicker.pick(context, [cameraPicker.PickerMediaType.PHOTO], profile);
      if (!result || result.resultCode !== 0 || result.resultUri.length === 0) {
        this.status = '未获取到照片';
        this.errorMessage = `camera result: ${result?.resultCode ?? 'empty'}`;
        return;
      }

      this.capturedImageUri = result.resultUri;
      this.sampleImageIndex = -1;
      this.status = '已获取照片，正在推理';
      await this.runDemoInference();
    } catch (error) {
      this.status = '拍照失败';
      this.errorMessage = `${error}`;
    }
  }

  private async useSampleImage(): Promise<void> {
    this.capturedImageUri = '';
    this.sampleImageIndex = (this.sampleImageIndex + 1) % 3;
    this.status = `已加载样例 ${this.sampleImageIndex + 1}，正在推理`;
    await this.runDemoInference();
  }

  private async runDemoInference(): Promise<void> {
    this.status = this.capturedImageUri.length > 0 || this.sampleImageIndex >= 0 ? '正在解码图片' : '正在加载端侧模型';
    this.errorMessage = '';
    this.predictions = [];
    this.recognizedText = '';
    this.confidence = '--';
    this.elapsedMs = '--';

    try {
      const manager = getContext(this).resourceManager;
      const decodedImage = await this.loadCurrentImagePixels();
      this.status = '正在执行 ncnn 推理';

      const recognizer = nativeRecognizer as NativeRecognizer;
      let rawResult: string;
      if (decodedImage) {
        const paramBytes = await manager.getRawFileContent('ppocrv5_mobile_rec.param');
        const modelBytes = await manager.getRawFileContent('ppocrv5_mobile_rec.bin');
        const labelBytes = await manager.getRawFileContent('ppocrv5_dict.txt');
        rawResult = recognizer.recognize(
          toArrayBuffer(paramBytes),
          toArrayBuffer(modelBytes),
          decodedImage.pixels,
          decodedImage.width,
          decodedImage.height,
          toArrayBuffer(labelBytes)
        );
      } else {
        const paramBytes = await manager.getRawFileContent('hwdb_demo.param');
        const modelBytes = await manager.getRawFileContent('hwdb_demo.bin');
        rawResult = recognizer.recognize(toArrayBuffer(paramBytes), toArrayBuffer(modelBytes));
      }
      const result = JSON.parse(rawResult) as NativeResult;
      if (!result.ok) {
        this.status = '推理失败';
        this.errorMessage = result.error ?? 'unknown native error';
        return;
      }

      this.status = '推理完成';
      this.elapsedMs = `${result.elapsedMs ?? 0} ms`;
      this.recognizedText = result.text ?? '';
      this.confidence = result.confidence !== undefined ? `${(result.confidence * 100).toFixed(2)}%` : '--';
      this.predictions = result.topK ?? [];
    } catch (error) {
      this.status = '推理异常';
      this.errorMessage = `${error}`;
    }
  }

  private async loadCurrentImagePixels(): Promise<DecodedImage | undefined> {
    if (this.capturedImageUri.length > 0) {
      const source = image.createImageSource(this.capturedImageUri);
      return await this.decodeImageSource(source);
    }

    if (this.sampleImageIndex >= 0) {
      const manager = getContext(this).resourceManager;
      const sampleBytes = await manager.getRawFileContent(`sample_handwriting_${this.sampleImageIndex + 1}.png`);
      const source = image.createImageSource(toArrayBuffer(sampleBytes));
      return await this.decodeImageSource(source);
    }

    return undefined;
  }

  private async decodeImageSource(source: image.ImageSource): Promise<DecodedImage> {
    const options: image.DecodingOptions = {
      desiredPixelFormat: image.PixelMapFormat.RGBA_8888
    };
    const pixelMap = await source.createPixelMap(options);
    const info = await pixelMap.getImageInfo();
    const width = info.size.width;
    const height = info.size.height;
    const pixels = new ArrayBuffer(width * height * 4);
    await pixelMap.readPixelsToBuffer(pixels);
    await pixelMap.release();
    await source.release();
    return { pixels, width, height };
  }

  aboutToAppear(): void {
    this.runDemoInference();
  }

  build() {
    Scroll() {
      Column({ space: 18 }) {
        Row() {
          Column({ space: 4 }) {
            Text('手写汉字识别')
              .fontSize(26)
              .fontWeight(FontWeight.Bold)
            Text('拍照采集 + ArkTS + Native C++ + ncnn')
              .fontSize(14)
              .fontColor('#5f6673')
          }
          .alignItems(HorizontalAlign.Start)
          .layoutWeight(1)
        }
        .width('100%')
        .alignItems(VerticalAlign.Center)

        Row({ space: 10 }) {
          Button('拍照识别')
            .height(42)
            .layoutWeight(1)
            .onClick(() => {
              this.capturePhotoAndRecognize();
            })

          Button('重跑模型')
            .height(42)
            .layoutWeight(1)
            .onClick(() => {
              this.runDemoInference();
            })

          Button('样例图片')
            .height(42)
            .layoutWeight(1)
            .onClick(() => {
              this.useSampleImage();
            })
        }
        .width('100%')

        Column({ space: 10 }) {
          if (this.capturedImageUri.length > 0) {
            Image(this.capturedImageUri)
              .width('100%')
              .height(220)
              .objectFit(ImageFit.Contain)
              .backgroundColor('#eef1f5')
              .borderRadius(8)
          } else if (this.sampleImageIndex === 0) {
            Image($r('app.media.sample_handwriting_1'))
              .width('100%')
              .height(220)
              .objectFit(ImageFit.Contain)
              .backgroundColor('#eef1f5')
              .borderRadius(8)
          } else if (this.sampleImageIndex === 1) {
            Image($r('app.media.sample_handwriting_2'))
              .width('100%')
              .height(220)
              .objectFit(ImageFit.Contain)
              .backgroundColor('#eef1f5')
              .borderRadius(8)
          } else if (this.sampleImageIndex === 2) {
            Image($r('app.media.sample_handwriting_3'))
              .width('100%')
              .height(220)
              .objectFit(ImageFit.Contain)
              .backgroundColor('#eef1f5')
              .borderRadius(8)
          } else {
            Column() {
              Text('尚未拍照或选择样例')
                .fontSize(16)
                .fontColor('#6b7280')
            }
            .width('100%')
            .height(160)
            .justifyContent(FlexAlign.Center)
            .backgroundColor('#eef1f5')
            .borderRadius(8)
          }
        }
        .width('100%')

        Column({ space: 12 }) {
          Text(this.status)
            .fontSize(18)
            .fontWeight(FontWeight.Medium)
          Text(`耗时：${this.elapsedMs}`)
            .fontSize(14)
            .fontColor('#4b5563')
          if (this.recognizedText.length > 0) {
            Text(`识别结果：${this.recognizedText}`)
              .fontSize(22)
              .fontWeight(FontWeight.Bold)
              .fontColor('#111827')
            Text(`置信度：${this.confidence}`)
              .fontSize(14)
              .fontColor('#4b5563')
          }

          if (this.errorMessage.length > 0) {
            Text(this.errorMessage)
              .fontSize(13)
              .fontColor('#b42318')
          }
        }
        .width('100%')
        .padding(16)
        .borderRadius(8)
        .backgroundColor('#f4f6f8')
        .alignItems(HorizontalAlign.Start)

        Column({ space: 10 }) {
          Text(this.predictions.length > 0 ? 'Top-5 结果' : 'OCR 输出')
            .fontSize(18)
            .fontWeight(FontWeight.Medium)
            .width('100%')

          ForEach(this.predictions, (item: NativePrediction) => {
            Row() {
              Text(LABELS[item.index] ?? `${item.index}`)
                .fontSize(24)
                .fontWeight(FontWeight.Bold)
                .width(56)
              Progress({
                value: Math.round(item.score * 1000),
                total: 1000,
                type: ProgressType.Linear
              })
                .layoutWeight(1)
                .height(8)
              Text(`${(item.score * 100).toFixed(2)}%`)
                .fontSize(13)
                .fontColor('#4b5563')
                .width(72)
                .textAlign(TextAlign.End)
            }
            .width('100%')
            .height(44)
          }, (item: NativePrediction) => `${item.index}`)

          if (this.predictions.length === 0 && this.errorMessage.length === 0) {
            Text(this.recognizedText.length > 0 ? this.recognizedText : '暂无结果')
              .fontSize(14)
              .fontColor('#6b7280')
              .width('100%')
          }
        }
        .width('100%')
        .alignItems(HorizontalAlign.Start)

        Text('已接入 PP-OCRv5 mobile rec ncnn 模型；拍照或样例图片会解码为 RGBA 后传入 native 预处理、推理并进行 CTC 解码。')
          .fontSize(13)
          .fontColor('#6b7280')
          .lineHeight(20)
          .width('100%')
      }
      .width('100%')
      .padding({ left: 20, right: 20, top: 28, bottom: 20 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#ffffff')
  }
}
