import argparse
import json
import sys

from src.consumer import ConsumerKafka, ConsumerFile


def main():
    parser = argparse.ArgumentParser(description="consumer")

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config1.json",
        help=u"Config file located in ./config/ directory."
    )

    """parser.add_argument(
        "-f",
        "--fit",
        dest="train_data",
        help=u"Train the model from .csv data file in ./data/train/ directory."
    )

    parser.add_argument(
        "-s",
        "--save",
        dest="model_file_name_save",
        help=u"Save the model to a file in directory ./models/"
    )

    parser.add_argument(
        "-l",
        "--load",
        dest="model_file_name_load",
        help=u"Load the model from a file in directory ./models/"
    )"""

    parser.add_argument(
        "--file",
        dest="data_to_process",
        help=u"Read data from a specified file on specified location."
    )

    # Display help if no arguments are defined
    if (len(sys.argv) == 1):
        parser.print_help()
        sys.exit(1)

    # Parse input arguments
    args = parser.parse_args()

    if(args.data_to_process is not None):
        if(args.data_to_process[-3:] == "csv"):
            file_type = "csv"
        elif(args.data_to_process[-4:] == "json"):
            file_type = "json"
        else:
            print("Unknown data file type.")
            sys.exit(1)
        consumer = ConsumerFile(configuration_location=args.config,
                                file_type=file_type)
    else:
        consumer = ConsumerKafka(configuration_location=args.config)

    """if(args.train_data is not None):
        # fitting the models for anomaly detection algorithms that use them
        consumer.anomaly.fit(train_data=args.train_data)

    if(args.model_file_name_save is not None):
        consumer.anomaly.save(dest=args.model_file_name_save)

    if(args.model_file_name_load is not None):
        consumer.anomaly.load(dest=args.model_file_name_load)"""

    consumer.read()


if (__name__ == '__main__'):
    main()
