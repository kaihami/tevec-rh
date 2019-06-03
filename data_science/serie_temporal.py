#!/usr/bin/env python3

import json
import pickle
from fbprophet import Prophet
import pandas as pd
from collections import defaultdict
from tabulate import tabulate
import argparse

def parse_cli():
    parser = argparse.ArgumentParser(description = 'Forecast sales of SKU 2')

    parser.add_argument('-i',
                        '--input',
                        help = 'Input json file')

    parser.add_argument('-t',
                        '--test',
                        help = 'Testing this program using "serie_vendas_input.json" file',
                        action = 'store_true')

    parser.add_argument('-of',
                        '--output_format',
                        default = 'json',
                        help = 'Output formart (default: json) [table, df, json, sdtout]')

    parser.add_argument('-f',
                        '--file',
                        default = '',
                        help = 'File name to save the json')

    args = parser.parse_args()

    return args

class ForecastService:
    """
    ... Develop our code ...
    """

    def __init__(self):
        """
        ... Just an example ...
        """
        self.model = self.load_model()

    def forecast(self, json_file):
        """
        ... Develop our code ...
        """

        future = pd.DataFrame({'ds': self.load_json(json_file)})


        result = self.model.predict(future)
        result = result[['ds', 'yhat']]
        result.columns = ['date', 'forecast']
        result['forecast'] = result['forecast'].map(lambda x: round(x,0))
        result['forecast'] = result['forecast'].astype(int)
        self.result = result


    def load_json(self, j):
        loaded = json.load(open(j))
        return loaded['request']['dates']
    def load_model(self):
        with open('sku2.forecasting.Prophet.pkl', 'rb') as f:
            return pickle.load(f)

    def write(self, output_format = 'json', file_name = ''):
        if output_format == 'json':
            output = defaultdict(list)
            for i, row in self.result.iterrows():
                output['response'].append({'date': row.date.strftime("%Y-%m-%d"),
                                           'forecast': row.forecast})

            if file_name:
                with open(file_name, 'w') as outfile:
                    json.dump(output, outfile, indent=4)
            else:
                print(json.dumps(output, indent = 4))

        if output_format in ['table', 'df', 'stdout']:
            res = self.result
            res['date'] = res['date'].dt.strftime('%Y-%m-%d')
            print(tabulate(self.result, headers='keys', tablefmt='psql'))

if __name__ == "__main__":

    args = parse_cli()
    api = ForecastService()
    if args.test:


        api.forecast('./serie_vendas_input.json')
        api.write('json')

    else:
        api.forecast(args.input)
        api.write(args.output_format, args.file)
