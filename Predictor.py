from numpy.core.fromnumeric import shape
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from models.ResNet import ResNet
import json
from kafka import KafkaProducer, KafkaConsumer

KAFKA_HOST = 'localhost:9092'
FTOPICS = ['forward']
BTOPICS = ['backward']
producer = KafkaProducer(bootstrap_servers=KAFKA_HOST)


class params:
    data_dir = "/home/dev/classification/data"
    out_channels = 10
    num_workers = 4
    val_split = 0.8
    loss = "cross_entropy"
    max_nb_epochs = 100
    batch_size = 32
    aug = True


with torch.no_grad():
    # Loading the saved model
    ckpt_path = '/home/dev/classification/logs/loss_fn_cross_entropy_100Epochs_TrainRun/epoch=36-step=55499.ckpt'
    model = ResNet(params)
    model.load_model_weights_from_ckpt(ckpt_path)
    model.eval()


def publish_prediction(pred, request_id):
    global producer
    packet = {'request_id': str(request_id), 'prediction': pred}
    producer.send('backward', json.dumps(packet).encode('utf-8'))
    print(
        "\033[1;31;40m -- PRODUCER_back: Sent prediction of id {}".format(request_id))
    producer.flush()


def single_inference(image):
    '''Predicts class of input image, input format in integer list'''
    # responses =[]
    # for image in images:

    image = (np.array(image)).reshape((28, 28))

    image = np.stack([image, image, image], axis=2)
    image = Image.fromarray(np.uint8(image))
    trans = transforms.ToTensor()
    image = trans(image)

    image = image.unsqueeze(dim=0)
    #     responses.append(image)

    # batch = torch.cat(responses, dim = 0)
    # Generate prediction
    prediction = model(image)

    # Predicted class value using argmax
    predicted_class = (torch.argmax(prediction)).detach().numpy()
    return int(predicted_class)


def start():
    for msg in consumer:
        message = json.loads(msg.value)
        # print(message.keys())
        img = message.get('data')
        request_id = message.get('request_id')
        pred = single_inference(img)
        publish_prediction(pred, request_id)


if __name__ == '__main__':

    consumer = KafkaConsumer(bootstrap_servers=KAFKA_HOST)
    consumer.subscribe(FTOPICS)

    start()
