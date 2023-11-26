"""
Specific to intersection but works for all classic and learning based controllers
"""

import os 
import argparse

import numpy as np
import pandas as pd

class EvalMetrics():
    def __init__(self, args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def efficiency(self, ):
        """
        Throughput and Fuel consumption
        """
        
        pass

    def safety(self, ):
        """
        Time to Collision and Deceleration rate to avoid a crash
        For both, worst case taken
        Also for both, only control vehicles considered
        """
        pass

    def stability(self, ):
        """
        Only Controller acceleration variation here
        WAR has its own process to measure
        """
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating metrics for the agent')
    parser.add_argument('--emissions_file_path', type=str, default='./test_time_rollout',
                    help='Path to emissions file')
    parser.add_argument('--method', type=str, default=None)