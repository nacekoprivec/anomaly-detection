import argparse
import json
import sys
import requests
import threading
import time
import logging

from src.consumer import ConsumerKafka, ConsumerFile, ConsumerFileKafka


def ping_watchdog():
    interval = 30 # ping interval in seconds
    url = "atena.ijs.si"
    port = 5001
    path = "/pingCheckIn/Anomaly detection"

    # Continue sending pings only if main thread is still alive
    if(threading.main_thread().is_alive()):
        try:
            r = requests.get("http://{}:{}{}".format(url, port, path))
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            logging.warning(e)
        else:
            pass
            #logging.info('Successful ping at ' + time.ctime())
        threading.Timer(interval, ping_watchdog).start()

def main():
    logging.basicConfig(filename="event_log.log", format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    parser = argparse.ArgumentParser(description="consumer")

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config1.json",
        help=u"Config file located in ./config/ directory."
    )

    parser.add_argument(
        "-f",
        "--file",
        dest="data_file",
        action="store_true",
        help=u"Read data from a specified file on specified location."
    )

    parser.add_argument(
        "-fk",
        "--filekafka",
        dest="data_both",
        action="store_true",
        help=u"Read data from a specified file on specified location and then from kafka stream."
    )

    parser.add_argument(
        "-w",
        "--watchdog",
        dest="watchdog",
        action='store_true',
        help=u"Ping watchdog",
    )

    # Display help if no arguments are defined
    if (len(sys.argv) == 1):
        parser.print_help()
        sys.exit(1)

    # Parse input arguments
    args = parser.parse_args()

    if(args.data_file):   
        consumer = ConsumerFile(configuration_location=args.config)
    elif(args.data_both):
        consumer = ConsumerFileKafka(configuration_location=args.config)
    else:
        consumer = ConsumerKafka(configuration_location=args.config)

    # Ping watchdog every 30 seconds if specfied
    if (args.watchdog):
        print("=== Watchdog started ==", flush=True) 
        ping_watchdog()

    print("=== Service starting ===", flush=True)

    consumer.read()


if (__name__ == '__main__'):
    main()
