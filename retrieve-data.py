#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:45:03 2020

@author: jaisi8631
"""

import argparse
from yahoo_fin import stock_info

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--stock", type = str, default = "nflx",
                help = "ticker for stock")
args = vars(ap.parse_args())

stock_df = stock_info.get_data(args["stock"])
stock_df.index.names = ['date']


stock_df = stock_df.drop('ticker', 1)
stock_df = stock_df.drop('adjclose', 1)

stock_df.head()

stock_df.to_csv("stock_data.csv")