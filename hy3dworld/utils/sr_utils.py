import os
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer


# build sr model
def build_sr_model(scale=2, model_name=None, tile=0, tile_pad=10, pre_pad=0, fp32=False, gpu_id=None):
    # if model_name not specified, use default mapping
    if model_name is None:
        if scale == 2:
            model_name = 'RealESRGAN_x2plus'
        else:
            model_name = 'RealESRGAN_x4plus'

    # model architecture configs
    model_configs = {
        'RealESRGAN_x2plus': {
            'internal_scale': 2,
            'model': lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        },
        'RealESRGAN_x4plus': {
            'internal_scale': 4,
            'model': lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        }
    }

    if model_name not in model_configs:
        raise ValueError(
            f'Unknown model name: {model_name}. Available models: {list(model_configs.keys())}')

    config = model_configs[model_name]
    model = config['model']()
    file_url = [config['url']]

    model_path = os.getenv("HUNYUANWORLD_SR_WEIGHTS") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # restorer
    upsampler = RealESRGANer(
        scale=config['internal_scale'],  # Use the internal scale of the model
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id)

    return upsampler


# sr inference code
def sr_inference(input, output_path, upsampler, scale=2, ext='auto', suffix='sr'):
    os.makedirs(output_path, exist_ok=True)

    path = input
    imgname, extension = os.path.splitext(os.path.basename(path))

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    width = img.shape[1]

    # pad the image to make eliminate the border artifacts
    pad_len = width // 4
    img = cv2.copyMakeBorder(img, 0, 0, pad_len, pad_len, cv2.BORDER_WRAP)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None

    try:
        output, _ = upsampler.enhance(
            img, outscale=scale)  # Use the input scale as the final output amplification factor
        # remove the padding
        output = output[:, int(pad_len*scale):int((width+pad_len)*scale), :]
    except RuntimeError as error:
        print('Error', error)
        print(
            'If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        if ext == 'auto':
            extension = extension[1:]
        else:
            extension = ext
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        if suffix == '':
            save_path = os.path.join(output_path, f'{imgname}.{extension}')
        else:
            save_path = os.path.join(
                output_path, f'{imgname}_{suffix}.{extension}')
        cv2.imwrite(save_path, output)
