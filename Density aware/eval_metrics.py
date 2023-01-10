"""
Three metrics each for Safety, Efficiency and Stability
"""
import argparse
import numpy as np
import pandas as pd


class EvalMetrics():
    def __init__(self, args):
        self.args = args
        self.emissions_file_path = self.args.emissions_file_path

        dataframe = pd.read_csv(self.emissions_file_path)
        print(dataframe.head())

    def safety(self, ):
        pass 

    def efficiency(self, ):
        pass

    def stability(self, ):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating metrics for the agent')

    parser.add_argument('--emissions_file_path', type=str, 
                        help='Path to emissions file')


    metrics = EvalMetrics(parser.parse_args())

    metrics.safety()
    metrics.efficiency()
    metrics.stability()