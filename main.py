import argparse
import json
import sys

from src.consumer import ConsumerKafka, ConsumerFile, ConsumerFileKafka


def main():
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

    consumer.read()


if (__name__ == '__main__'):
    main()
