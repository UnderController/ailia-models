import cv2
import os
import sys
import ailia
import numpy as np
import time
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# import original modules
sys.path.append('../../util')
from image_utils import load_image, get_image_shape  # noqa: E402
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
import webcamera_utils  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

import torch

# ======================
# Parameters 1
# ======================

IMAGE_PATH1 = 'input1.png'
IMAGE_PATH2 = 'input2.png'
SAVE_IMAGE_PATH = 'output.png'
#REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/cain/'


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('CAIN model', IMAGE_PATH1, SAVE_IMAGE_PATH)
parser.add_argument('--inputs', nargs='*', default=['input1.png', 'input2.png'])
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================

WEIGHT_PATH = 'cain.onnx'
MODEL_PATH = 'cain.onnx.prototxt'


# ======================
# Main functions
# ======================


def quantize(img, rgb_range=255):
    return img.mul(255 / rgb_range).clamp(0, 255).round()

def recognize_from_image():

    out = np.load('out.npy')[0].transpose(1, 2, 0) * 255
    print(out[0][0])
    print(out.shape)
    cv2.imwrite('out.png', out)

    if args.onnx:
        import onnxruntime
        session = onnxruntime.InferenceSession('cain.onnx')
        model = onnxruntime.InferenceSession(WEIGHT_PATH)
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=-1)
    if args.inputs:
        logger.info(args.inputs)
        IMAGE_HEIGHT, IMAGE_WIDTH = get_image_shape(args.inputs[0])
    else:
        logger.info(IMAGE_PATH1, IMAGE_PATH2)
        IMAGE_HEIGHT, IMAGE_WIDTH = get_image_shape(IMAGE_PATH1) 

    net.set_input_shape((1, 3, IMAGE_HEIGHT,IMAGE_WIDTH))

    for iternum, image_path in enumerate(args.inputs):
        if  iternum == 0:
            img1 = image_path
            continue
        img2 = image_path

        # prepare input data
        logger.info(image_path)
        """
        input_data1 = np.zeros((4, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        input_data2 = np.zeros((4, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        for idx in range(4):
           input_data1[idx] = cv2.imread(img1).transpose((2, 0, 1))
           input_data2[idx] = cv2.imread(img2).transpose((2, 0, 1))
           """
        input_data1 = np.zeros((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        input_data2 = np.zeros((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        input_data1[0] = cv2.imread(img1).transpose((2, 0, 1))
        input_data2[0] = cv2.imread(img2).transpose((2, 0, 1))
        
        """
        input_data1 = torch.load('tensor1.pt')
        input_data2 = torch.load('tensor2.pt')
        q_im = quantize(input_data1.data.mul(255))
        im = Image.fromarray(q_im.permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
        im.save('before.png')
        print(input_data1)
        """
        #input_data1 = np.load('im1.npy')
        #input_data2 = np.load('im2.npy')
        #print(input_data1.shape)
        #print(cv2.imread(img1).shape)
        #output_img = Image.fromarray(input_data1[0].transpose((1, 2, 0)).astype(np.uint8))
        #output_img.save('before.png')

        """
        input_data1 = load_image(
            img1,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            gen_input_ailia=True,
            normalize_type='None'
        )
        input_data2 = load_image(
            img2,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            gen_input_ailia=True,
            normalize_type='None'
        )
        """
        #print(input_data2.shape)
        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                if args.onnx:
                    ort_inputs = {model.get_inputs()[0].name:input_data1, model.get_inputs()[1].name:input_data2}
                    model_out = model.run(None, ort_inputs)[0]
                else:
                    preds_ailia = net.run([input_data1, input_data2])[0]
                    end = int(round(time.time() * 1000))
                    logger.info(f'\tailia processing time {end - start} ms')
        else:
            if args.onnx:
                ort_inputs = {model.get_inputs()[0].name:input_data1, model.get_inputs()[1].name:input_data2}
                model_out = model.run(None, ort_inputs)[0]
                if iternum == 1:
                    const = 1
                    float_path = 0.5
                else:
                    const += 1
                
                output_img = model_out[0].transpose((1, 2, 0)) 
                #output_img = np.clip(output_img, 0, 255)
                print(output_img[0][0])
                print(output_img.shape)
                savepath = 'output_' + str(const+float_path) + '.png'
                logger.info(f'saved at : {savepath}')
                output_img = Image.fromarray(output_img.astype(np.uint8))
                output_img.save(savepath)

                #todo 冗長かつインプットのサイズが合わない
                """
                print(input_data1.shape)
                print(model_out[0].shape)
                ort_inputs = {model.get_inputs()[0].name:input_data1, model.get_inputs()[1].name:model_out[0]}
                model_out1 = model.run(None, ort_inputs)[0] 
                ort_inputs = {model.get_inputs()[0].name:model_out[0], model.get_inputs()[1].name:input_data2}
                model_out2 = model.run(None, ort_inputs)[0] 
                output_img = model_out1[0].transpose((1, 2, 0))
                output_img = np.clip(output_img, 0, 255)
                savepath = 'output_' + str(const+float_path/2) + '.png'
                logger.info(f'saved at : {savepath}')
                output_img = Image.fromarray(output_img.astype(np.uint8))
                output_img.save(savepath)
                output_img = model_out2[0].transpose((1, 2, 0))
                output_img = np.clip(output_img, 0, 255)
                savepath = 'output_' + str(const+float_path/2+float_path) + '.png'
                logger.info(f'saved at : {savepath}')
                output_img = Image.fromarray(output_img.astype(np.uint8))
                output_img.save(savepath)
                """


            else:
                #print(net.get_input_shape())
                #print(input_data1.shape)
                model_out = net.run([input_data1, input_data2])[0]

        # postprocessing
        """
        for _ in range(len(args.inputs)-1):
            if iternum == 1:
                const = 1
            else:
                const += 1
            for j in range(len(model_out)):
                const += 0.25
                output_img = model_out[0].transpose((1, 2, 0))
                #output_img = model_out[1].reshape((1, 3, 256, 448))
                output_img = np.clip(output_img, 0, 255)
                print(output_img.shape)
                cv2.imwrite('sample.png', output_img)
                savepath = 'output_' + str(const) + '.png'
                logger.info(f'saved at : {savepath}')
                #print(output_img.shape)
                output_img = Image.fromarray(output_img.astype(np.uint8))
                output_img.save(savepath)
            """

        """
        output_img = preds_ailia.transpose(1, 2, 0)
        output_img = np.clip(output_img, 0, 255)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output_img)
        """
        img1 = img2
    logger.info('Script finished successfully.')



# todo after recognize_from_image done
def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=-1)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    time.sleep(1)  

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        IMAGE_HEIGHT, IMAGE_WIDTH = frame.shape[0], frame.shape[1]
        input_image, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='None'
        )
        net.set_input_shape((1,3,IMAGE_HEIGHT,IMAGE_WIDTH))

        # Inference
        preds_ailia = net.predict(input_data)[0] 
        output_img = preds_ailia.transpose(1, 2, 0) / 255
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', output_img)
        output_img *= 255
        output_img = np.clip(output_img, 0, 255)     
        #cv2.imwrite('pred.png', output_img) #please uncomment to save output

        # save results
        if writer is not None:
            writer.write(output_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    #check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    
    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
