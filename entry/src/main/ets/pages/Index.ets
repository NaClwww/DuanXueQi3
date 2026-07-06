import abilityAccessCtrl from '@ohos.abilityAccessCtrl';
import type { Permissions } from '@ohos.abilityAccessCtrl';
import camera from '@ohos.multimedia.camera';
import cameraPicker from '@ohos.multimedia.cameraPicker';
import image from '@ohos.multimedia.image';
import photoAccessHelper from '@ohos.file.photoAccessHelper';
import pasteboard from '@ohos.pasteboard';
import fs from '@ohos.file.fs';
import nativeRecognizer from 'libentry.so';

interface NativeRecognizer {
  recognize(
    param: ArrayBuffer,
    model: ArrayBuffer,
    rgba?: ArrayBuffer,
    width?: number,
    height?: number,
    labels?: ArrayBuffer,
    detParam?: ArrayBuffer,
    detModel?: ArrayBuffer,
    recParam?: ArrayBuffer,
    recModel?: ArrayBuffer,
    useGpu?: boolean
  ): string;

  getRuntimeInfo(useGpu?: boolean): string;
}

interface NativePrediction {
  index: number;
  score: number;
}

interface NativeBox {
  x: number;
  y: number;
  w: number;
  h: number;
  detScore: number;
  text: string;
  confidence: number;
}

interface NativeResult {
  ok: boolean;
  elapsedMs?: number;
  topK?: NativePrediction[];
  text?: string;
  confidence?: number;
  boxes?: NativeBox[];
  detError?: string;
  error?: string;
}

interface DecodedImage {
  pixels: ArrayBuffer;
  width: number;
  height: number;
}

const LABELS: string[] = [];

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

@Entry
@Component
struct Index {
  @State status: string = '等待选择图片';
  @State elapsedMs: string = '--';
  @State predictions: NativePrediction[] = [];
  @State ocrBoxes: NativeBox[] = [];
  @State selectedBoxIndexes: number[] = [];
  @State isRecognizing: boolean = false;
  @State recognizedText: string = '';
  @State confidence: string = '--';
  @State errorMessage: string = '';
  @State debugInfo: string = '';
  @State useGpu: boolean = false;
  @State showDebugPage: boolean = false;
  @State capturedImageUri: string = '';
  @State selectedImageUri: string = '';

  private async capturePhotoAndRecognize(): Promise<void> {
    this.status = '姝ｅ湪鎵撳紑鐩告満';
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
        this.status = '鏈幏鍙栧埌鐓х墖';
        this.errorMessage = `camera result: ${result?.resultCode ?? 'empty'}`;
        return;
      }

      this.capturedImageUri = result.resultUri;
      this.selectedImageUri = '';
      this.status = '宸茶幏鍙栫収鐗囷紝姝ｅ湪鎺ㄧ悊';
      await this.runDemoInference();
    } catch (error) {
      this.status = '鎷嶇収澶辫触';
      this.errorMessage = `${error}`;
    }
  }

  private async chooseImageAndRecognize(): Promise<void> {
    this.status = '姝ｅ湪鎵撳紑鍥惧簱';
    this.errorMessage = '';

    try {
      const options = new photoAccessHelper.PhotoSelectOptions();
      options.MIMEType = photoAccessHelper.PhotoViewMIMETypes.IMAGE_TYPE;
      options.maxSelectNumber = 1;
      const picker = new photoAccessHelper.PhotoViewPicker();
      const result = await picker.select(options);
      if (!result || result.photoUris.length === 0) {
        this.status = '鏈€夋嫨鍥剧墖';
        return;
      }

      this.capturedImageUri = '';
      this.selectedImageUri = result.photoUris[0];
      this.status = '已选择图片，正在推理';
      await this.runDemoInference();
    } catch (error) {
      this.status = '閫夋嫨鍥剧墖澶辫触';
      this.errorMessage = `${error}`;
    }
  }

  private showRuntimeInfo(): void {
    try {
      const recognizer = nativeRecognizer as NativeRecognizer;
      const rawInfo = recognizer.getRuntimeInfo(this.useGpu);
      const info = JSON.parse(rawInfo) as Record<string, Object>;
      this.debugInfo =
        `backend: ${info['inferenceBackend']}\n` +
        `useVulkanCompute: ${info['useVulkanCompute']}\n` +
        `vulkanCompiled: ${info['vulkanCompiled']}\n` +
        `gpuCount: ${info['gpuCount']}\n` +
        `gpuInstanceRet: ${info['gpuInstanceRet']}\n` +
        `ncnn: ${info['ncnnVersion']}\n` +
        `threads: ${info['numThreads']}`;
    } catch (error) {
      this.debugInfo = `${error}`;
    }
  }

  private isBoxSelected(index: number): boolean {
    return this.selectedBoxIndexes.indexOf(index) >= 0;
  }

  private toggleBoxSelection(index: number): void {
    const selected = this.selectedBoxIndexes.slice();
    const position = selected.indexOf(index);
    if (position >= 0) {
      selected.splice(position, 1);
    } else {
      selected.push(index);
    }
    this.selectedBoxIndexes = selected;
  }

  private copySelectedBoxText(): void {
    if (this.selectedBoxIndexes.length === 0) {
      this.errorMessage = '请先选择识别块';
      return;
    }
    const text = this.selectedBoxIndexes
      .slice()
      .sort((left: number, right: number) => left - right)
      .map((index: number) => this.ocrBoxes[index]?.text ?? '')
      .filter((item: string) => item.length > 0)
      .join('\n');
    if (text.length === 0) {
      this.errorMessage = '閫変腑鐨勮瘑鍒潡娌℃湁鏂囨湰';
      return;
    }

    try {
      const pasteData = pasteboard.createData(pasteboard.MIMETYPE_TEXT_PLAIN, text);
      const systemPasteboard = pasteboard.getSystemPasteboard();
      systemPasteboard.setData(pasteData);
      this.errorMessage = `宸插鍒讹細\n${text}`;
    } catch (error) {
      this.errorMessage = `复制失败：${error}`;
    }
  }

  private async runDemoInference(): Promise<void> {
    this.status = this.capturedImageUri.length > 0 || this.selectedImageUri.length > 0 ? '正在解码图片' : '等待选择图片';
    this.isRecognizing = true;
    this.errorMessage = '';
    this.predictions = [];
    this.ocrBoxes = [];
    this.selectedBoxIndexes = [];
    this.recognizedText = '';
    this.confidence = '--';
    this.elapsedMs = '--';

    try {
      const manager = getContext(this).resourceManager;
      const decodedImage = await this.loadCurrentImagePixels();
      this.status = '姝ｅ湪鎵ц ncnn 鎺ㄧ悊';

      const recognizer = nativeRecognizer as NativeRecognizer;
      let rawResult: string;
      if (decodedImage) {
        const detParamBytes = await manager.getRawFileContent('ppocrv6_small_det.param');
        const detModelBytes = await manager.getRawFileContent('ppocrv6_small_det.bin');
        const recParamBytes = await manager.getRawFileContent('ppocrv6_small_rec.param');
        const recModelBytes = await manager.getRawFileContent('ppocrv6_small_rec.bin');
        const labelBytes = await manager.getRawFileContent('ppocrv6_dict.txt');
        rawResult = recognizer.recognize(
          toArrayBuffer(detParamBytes),
          toArrayBuffer(detModelBytes),
          decodedImage.pixels,
          decodedImage.width,
          decodedImage.height,
          toArrayBuffer(labelBytes),
          toArrayBuffer(detParamBytes),
          toArrayBuffer(detModelBytes),
          toArrayBuffer(recParamBytes),
          toArrayBuffer(recModelBytes),
          this.useGpu
        );
      } else {
        const paramBytes = await manager.getRawFileContent('hwdb_demo.param');
        const modelBytes = await manager.getRawFileContent('hwdb_demo.bin');
        rawResult = recognizer.recognize(toArrayBuffer(paramBytes), toArrayBuffer(modelBytes));
      }
      const result = JSON.parse(rawResult) as NativeResult;
      if (!result.ok) {
        this.status = '鎺ㄧ悊澶辫触';
        this.errorMessage = rawResult;
        this.isRecognizing = false;
        return;
      }

      this.status = '鎺ㄧ悊瀹屾垚';
      this.isRecognizing = false;
      this.elapsedMs = `${result.elapsedMs ?? 0} ms`;
      this.recognizedText = result.text ?? '';
      this.confidence = result.confidence !== undefined ? `${(result.confidence * 100).toFixed(2)}%` : '--';
      this.predictions = result.topK ?? [];
      this.ocrBoxes = (result.boxes ?? []).filter((box: NativeBox) => box.confidence >= 0.2);
      this.selectedBoxIndexes = [];
      this.errorMessage = result.detError ?? '';
    } catch (error) {
      this.status = '鎺ㄧ悊寮傚父';
      this.isRecognizing = false;
      this.errorMessage = `${error}`;
    }
  }

  private async loadCurrentImagePixels(): Promise<DecodedImage | undefined> {
    if (this.capturedImageUri.length > 0) {
      const source = image.createImageSource(this.capturedImageUri);
      return await this.decodeImageSource(source);
    }

    if (this.selectedImageUri.length > 0) {
      const file = fs.openSync(this.selectedImageUri, fs.OpenMode.READ_ONLY);
      try {
        const source = image.createImageSource(file.fd);
        return await this.decodeImageSource(source);
      } finally {
        fs.closeSync(file);
      }
    }


    return undefined;
  }

  private async decodeImageSource(source: image.ImageSource): Promise<DecodedImage> {
    if (!source) {
      throw new Error('createImageSource failed');
    }
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

  @Builder
  DebugPage() {
    Scroll() {
      Column({ space: 16 }) {
        Row() {
          Text('Debug')
            .fontSize(26)
            .fontWeight(FontWeight.Bold)
            .layoutWeight(1)

          Button('杩斿洖')
            .height(38)
            .onClick(() => {
              this.showDebugPage = false;
            })
        }
        .width('100%')
        .alignItems(VerticalAlign.Center)

        Column({ space: 12 }) {
          Row() {
            Text('鍚敤 GPU')
              .fontSize(18)
              .fontWeight(FontWeight.Medium)
              .layoutWeight(1)
            Button(this.useGpu ? 'ON' : 'OFF')
              .width(88)
              .height(38)
              .onClick(() => {
                this.useGpu = !this.useGpu;
                this.showRuntimeInfo();
              })
          }
          .width('100%')
          .alignItems(VerticalAlign.Center)

          Button('刷新运行状态')
            .height(40)
            .width('100%')
            .onClick(() => {
              this.showRuntimeInfo();
            })

          Text(this.debugInfo.length > 0 ? this.debugInfo : '暂无运行状态')
            .fontSize(14)
            .fontColor('#374151')
            .lineHeight(20)
            .width('100%')
        }
        .width('100%')
        .padding(16)
        .borderRadius(8)
        .backgroundColor('#f4f6f8')
        .alignItems(HorizontalAlign.Start)

        Text('GPU 开关是请求启用 Vulkan；如果设备或模拟器没有可用 GPU，native 会自动回退 CPU。')
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

  build() {
    if (this.showDebugPage) {
      this.DebugPage()
    } else {
      Scroll() {
      Column({ space: 18 }) {
        Row() {
          Column({ space: 4 }) {
            Text('鎵嬪啓姹夊瓧璇嗗埆')
              .fontSize(26)
              .fontWeight(FontWeight.Bold)
          }
          .alignItems(HorizontalAlign.Start)
          .layoutWeight(1)

          Button('Debug')
            .height(28)
            .fontSize(12)
            .fontColor('#4b5563')
            .backgroundColor('#00000000')
            .padding({ left: 4, right: 4 })
            .border({ width: 0, color: '#00000000' })
            .onClick(() => {
              this.showRuntimeInfo();
              this.showDebugPage = true;
            })
        }
        .width('100%')
        .alignItems(VerticalAlign.Center)

        Row({ space: 10 }) {
          Button('鎷嶇収璇嗗埆')
            .height(42)
            .layoutWeight(1)
            .onClick(() => {
              this.capturePhotoAndRecognize();
            })

          Button('閲嶈窇妯″瀷')
            .height(42)
            .layoutWeight(1)
            .onClick(() => {
              this.runDemoInference();
            })


          Button('鍥惧簱閫夋嫨')
            .height(42)
            .layoutWeight(1)
            .onClick(() => {
              this.chooseImageAndRecognize();
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
          } else if (this.selectedImageUri.length > 0) {
            Image(this.selectedImageUri)
              .width('100%')
              .height(220)
              .objectFit(ImageFit.Contain)
              .backgroundColor('#eef1f5')
              .borderRadius(8)
          } else {
            Column() {
              Text('灏氭湭鎷嶇収鎴栭€夋嫨鏍蜂緥')
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

        if (false) {
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
            Text(`缃俊搴︼細${this.confidence}`)
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
        }

        Column({ space: 10 }) {
          Text('OCR 杈撳嚭')
            .fontSize(18)
            .fontWeight(FontWeight.Medium)
            .width('100%')

          if (this.isRecognizing) {
            Text('识别中...')
              .fontSize(15)
              .fontColor('#4b5563')
              .width('100%')
          }

          if (this.errorMessage.length > 0) {
            Text(this.errorMessage)
              .fontSize(13)
              .fontColor('#b42318')
              .width('100%')
          }

          if (this.ocrBoxes.length > 0) {
            Row() {
              Text(`分块结果：${this.ocrBoxes.length}`)
                .fontSize(14)
                .fontColor('#4b5563')
                .layoutWeight(1)
              Button('澶嶅埗閫変腑')
                .height(36)
                .onClick(() => {
                  this.copySelectedBoxText();
                })
            }
            .width('100%')
            .alignItems(VerticalAlign.Center)

            ForEach(this.ocrBoxes, (box: NativeBox, index?: number) => {
              Column({ space: 6 }) {
                Row() {
                  Text(this.isBoxSelected(index ?? -1) ? '√' : ' ')
                    .fontSize(16)
                    .width(24)
                    .textAlign(TextAlign.Center)
                  Text(box.text.length > 0 ? box.text : '(空)')
                    .fontSize(18)
                    .fontWeight(FontWeight.Medium)
                    .layoutWeight(1)
                    .maxLines(4)
                    .lineHeight(24)
                    .textOverflow({ overflow: TextOverflow.Ellipsis })
                }
                .width('100%')
                .alignItems(VerticalAlign.Center)

              }
              .width('100%')
              .padding(12)
              .borderRadius(8)
              .backgroundColor(this.isBoxSelected(index ?? -1) ? '#e5e7eb' : '#f4f6f8')
              .onClick(() => {
                this.toggleBoxSelection(index ?? -1);
              })
            }, (box: NativeBox, index?: number) => `${index ?? 0}-${box.x}-${box.y}-${box.text}`)
          }

          if (this.ocrBoxes.length === 0 && this.errorMessage.length === 0) {
            Text(this.isRecognizing ? '' : (this.recognizedText.length > 0 ? this.recognizedText : '暂无结果'))
              .fontSize(14)
              .fontColor('#6b7280')
              .width('100%')
              .height(120)
          }
        }
        .width('100%')
        .justifyContent(FlexAlign.Start)
        .alignItems(HorizontalAlign.Start)

      }
      .width('100%')
      .padding({ left: 20, right: 20, top: 28, bottom: 20 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#ffffff')
    }
  }
}
