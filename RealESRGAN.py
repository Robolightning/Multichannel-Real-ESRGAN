import cv2
import io
import os
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan.utils import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer

def RealESRGAN_upscaler(binary_data, args):
    # определяет модели в соответствии с выбранной моделью
    model_name = args["model"]
    num_in_ch = 3
    num_out_ch = num_in_ch
    seed_everything(args["seed"])
    
    if model_name == "RealESRGAN_x4plus":  # модель x4 RRDBNet
        model = RRDBNet(num_in_ch = num_in_ch, num_out_ch = num_out_ch, num_feat = 64, num_block = 23, num_grow_ch = 32, scale = 4)
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"]
    elif model_name == "RealESRNet_x4plus":  # модель x4 RRDBNet
        model = RRDBNet(num_in_ch = num_in_ch, num_out_ch = num_out_ch, num_feat = 64, num_block = 23, num_grow_ch = 32, scale = 4)
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"]
    elif model_name == "RealESRGAN_x4plus_anime_6B":  # модель x4 RRDBNet с 6 блоками
        model = RRDBNet(num_in_ch = num_in_ch, num_out_ch = num_out_ch, num_feat = 64, num_block = 6, num_grow_ch = 32, scale = 4)
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"]
    elif model_name == "RealESRGAN_x2plus":  # модель x2 RRDBNet
        model = RRDBNet(num_in_ch = num_in_ch, num_out_ch = num_out_ch, num_feat = 64, num_block = 23, num_grow_ch = 32, scale = 2)
        netscale = 2
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"]
    elif model_name == "realesr-animevideov3":  # модель x4 VGG-стиля (размера XS)
        model = SRVGGNetCompact(num_in_ch = num_in_ch, num_out_ch = num_out_ch, num_feat = 64, num_conv = 16, upscale = 4, act_type = "prelu")
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"]
    elif model_name == "realesr-general-x4v3":  # модель x4 VGG-стиля (размера S)
        model = SRVGGNetCompact(num_in_ch = num_in_ch, num_out_ch = num_out_ch, num_feat = 64, num_conv = 32, upscale = 4, act_type = "prelu")
        netscale = 2
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"]
    
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # путь к модели будет обновлён
            model_path = load_file_from_url(url = url, model_dir = os.path.join(ROOT_DIR, 'weights'), progress = True, file_name = None)

    # использовать dni для контроля силы удаления шума
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and args["denoise_strength"] != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args["denoise_strength"], 1 - args["denoise_strength"]]
    # восстановитель
    upsampler = RealESRGANer(scale = netscale, model_path = model_path, dni_weight = dni_weight, model = model, tile = args["tile"], tile_pad = args["tile_pad"], pre_pad = args["pre_pad"], half = not args["fp32"], gpu_id = args["gpu-id"])
    if args["face_enhance"]:  # Использовать GFPGAN для улучшения лиц
        if "RestoreFormer" in args["version"]:
            gfpgan_model_name = "RestoreFormer.ckpt"
            arch = args["version"]
            model_path = "RestoreFormer/weights/"
        else:
            gfpgan_model_list = ["GFPGANv1.pth", "GFPGANv1.3.pth", "GFPGANv1.4.pth"]
            gfpgan_model_name = "GFPGANCleanv1-NoCE-C2.pth"
            for mn in gfpgan_model_list:
                if args["version"] in mn:
                    gfpgan_model_name = mn
            arch = "clean"
            model_path = "gfpgan/weights/"
        face_enhancer = GFPGANer(model_path = model_path + gfpgan_model_name, upscale = args["outscale"], arch = arch, channel_multiplier = 2, bg_upsampler = upsampler, input_is_latent = args["input_is_latent"])
        img = cv2.cvtColor(np.array(Image.open(io.BytesIO(binary_data)).convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor(np.array(Image.open(io.BytesIO(binary_data)).convert("RGBA")), cv2.COLOR_RGBA2BGRA)
    try:
        if args["face_enhance"]:
            _, _, output = face_enhancer.enhance(img, has_aligned = False, only_center_face = False, paste_back = True)
        else:
            output, _ = upsampler.enhance(img, outscale = args["outscale"], num_out_ch = num_out_ch, alpha_upsampler = args["alpha_upsampler"])
    except RuntimeError as error:
        print("Ошибка", error, "\nЕсли у вас появляется ошибка \"CUDA out of memory\", попробуйте поставить параметру \"tile\" меньшее значение")
    else:
        im_buf_arr = cv2.imencode(".png", output)[1]
        result_binary_data = im_buf_arr.tobytes()
    torch.cuda.empty_cache()
    return result_binary_data

def get_img_type(i_name):
    img = Image.open(tp + "test\\" + filename).convert("RGBA")
    w, h = img.size
    bl = False
    wh = False
    o_clr = 0
    for i in range(w):
        for j in range(h):
            clr = img.getpixel((i, j))
            if clr[3] == 255:
                bl = True
            elif clr[3] == 0:
                wh = True
            else:
                if o_clr > 0:
                    return (0)
                o_clr += 1
                th = clr
    img.close()
    if o_clr == 0:
        return (bl, wh)
    return (bl, wh, th)

if __name__ == '__main__':
    params = {
            "model": "realesr-animevideov3",    #Модель для обработки ("RealESRGAN_x4plus" - модель x4 RRDBNet, "RealESRNet_x4plus" - модель x4 RRDBNet, "RealESRGAN_x4plus_anime_6B" - модель x4 RRDBNet с 6 блоками, "RealESRGAN_x2plus" - модель x2 RRDBNet, "realesr-animevideov3" - модель x4 VGG-стиля (размера XS), "realesr-general-x4v3" - модель x4 VGG-стиля (размера S)) 
            "denoise_strength": 0.0,            #Сила удаления шума. 0 для слабого удаления шума (шум сохраняется), 1 для сильного удаления шума. Используется только для модели "realesr-general-x4v3"
            "outscale": 2,                      #Величина того, во сколько раз увеличть разшрешение изображения (модель "RealESRGAN_x2plus" x2, остальные x4)
            "tile": 0,                          #Размер плитки, 0 для отсутствия плитки во время тестирования
            "tile_pad": 10,                     #Заполнение плитки
            "pre_pad": 0,                       #Предварительный размер заполнения на каждой границе
            "face_enhance": True,               #Использовать GFPGAN улучшения лиц
            "version": "RestoreFormer",         #Версия модели для улучшения лиц. Только если выбран "face_enhance: True. Возможне значения: "1.1", "1.2", "1.3", "1.4", "RestoreFormerGFPGAN", "RestoreFormer". Модель 1.1 тестовая, но способна колоризировать. Модель 1.2 обучена на большем количестве данных с предобработкой, не способна колоризировать, генерирует достаточно чёткие изображения с красивым магияжем, однако иногда результат генерации выглядит не натурально. Модель 1.3 основана на модели 1.2, генерирует более натурально выглядящие изображения, однако не такие чёткие, выдаёт лучие результаты на более низкокачественных изображениях, работает с относительно высококачественными изображениями, может иметь повторяющееся (дважды) восстановление. Модель 1.4 обеспечивает немного больше деталей и лучшую идентичность. Модель RestoreFormer создана специально для улучшения лиц, "RestoreFormer_GFPGAN" обеспечивает более чёткую, однако менее натуралистичную обработку и иногда создаёт артифакты.
            "input_is_latent": True,            #Скрытый ли вход. Только для Только если выбран "face_enhance: True и "version" от 1.1 до 1.4. Если выбран, то результат менее насыщенный и чёткий, но более наруральный
            "fp32": True,                       #Использовать точность fp32 во время вывода. По умолчанию fp16 (половинная точность)
            "alpha_upsampler": "realesrgan",    #Апсемплер для альфа-каналов. Варианты: "realesrgan" | "bicubic", Только для "face_enhance" == False
            "gpu-id": None,                     #Устройство gpu для использования (по умолчанию = None) может быть 0, 1, 2 для обработки на нескольких GPU
            "seed": 42,                         #Начальное инициализирующее значение
            #на данный момент "max_dim": pow(1024, 2) ((для всех моделей, кроме "RealESRGAN_x2plus") и "outscale": 4), и pow(2048, 2) (для модели "RealESRGAN_x2plus" и "outscale": 2)
        }
    
    with open("img.png", "rb") as f:
        init_img_binary_data = f.read()
    binary_data = RealESRGAN_upscaler(init_img_binary_data, params)
    Image.open(io.BytesIO(binary_data)).save("big.png")