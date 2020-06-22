import json
from collections import defaultdict

import boto3
import logging
from os import path
from flask import Flask, jsonify

from benchmarker.util import list_stacks

logger = logging.getLogger('benchmark_viewer')

HERE = path.dirname(__file__)

app = Flask('benchmark_viewer')
BUCKET = 'bigmech'
PREFIX = 'indra-db/benchmarks/'


@app.route('/', methods=['GET'])
def serve_page():
    with open(path.join(HERE, 'benchmark.html'), 'r') as f:
        return f.read().replace('{{ stacks }}', json.dumps(list_stacks()))


@app.route('/fetch/<corpus_name>/<stack_name>', methods=['GET'])
def get_stack_data(corpus_name, stack_name):
    try:
        s3 = boto3.client('s3')
        res = s3.list_objects_v2(Bucket=BUCKET,
                                 Prefix=f'{PREFIX}{corpus_name}/{stack_name}/')
        keys = {path.basename(e['Key']).split('.')[0]: e['Key']
                for e in res['Contents']}
        sorted_keys = list(sorted(keys.items(), key=lambda t: t[0],
                                  reverse=True))
        result = defaultdict(dict)
        for date_str, key in sorted_keys[:5]:
            date_str = path.basename(key).split('.')[0]
            file = s3.get_object(Bucket=BUCKET, Key=key)
            data = json.loads(file['Body'].read())
            for test_name, test_data in data.items():
                result[test_name][date_str] = test_data
    except Exception as e:
        logger.exception(e)
        return jsonify({'message': f'Error: {e}'}), 500
    return jsonify({'message': 'success', 'tests': result}), 200
