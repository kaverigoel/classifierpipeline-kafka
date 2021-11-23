from kafka import KafkaConsumer, KafkaProducer
import json
import argparse
import threading
from mnist import MNIST


KAFKA_HOST = 'localhost:9092'
FTOPICS = ['forward']
BTOPICS = ['backward']

parser = argparse.ArgumentParser(
    description='Running the unified API for Kafka')

parser.add_argument("--path",
                    type=str,
                    help="Path to file containing unzipped files to be inferenced")
parser.add_argument("--start_idx",
                    help="Start of index from which to inference from",
                    type=int)
parser.add_argument("--end_idx",
                    help="End of index from which to inference from",
                    type=int)
args = parser.parse_args()

PATH_TO_FILES = args.path

mndata = MNIST(PATH_TO_FILES)
# labels would not be needed for inferencing
test_images, test_labels = mndata.load_testing()


def start_producing():
    producer = KafkaProducer(bootstrap_servers=KAFKA_HOST)
    for i in range(args.start_idx, args.end_idx):
        message_id = str(i)
        message = {'request_id': message_id, 'data': test_images[i]}

        producer.send('forward', json.dumps(message).encode('utf-8'))

        print(
            "\033[1;31;40m -- PRODUCER_forward: Sent image with id {}".format(message_id))


def start_consuming():
    for msg in consumer:
        message = json.loads(msg.value)
        request_id = message.get('request_id')
        prediction = message.get('prediction')
        print(
            "\033[1;32;40m ** CONSUMER_back: Received predicted class {} for request id {}".format(prediction, request_id))


threads = []
consumer = KafkaConsumer(bootstrap_servers=KAFKA_HOST)
consumer.subscribe(BTOPICS)
t = threading.Thread(target=start_producing)
t2 = threading.Thread(target=start_consuming)
threads.append(t)
threads.append(t2)
t.start()
t2.start()
