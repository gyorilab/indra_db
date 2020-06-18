import json
import boto3
from os import path
from flask import Flask, jsonify

HERE = path.dirname(__file__)

app = Flask('benchmark_viewer')
BUCKET = 'bigmech'
BASE = 'indra-db/benchmarks'


@app.route('/', methods=['GET'])
def serve_page():
    with open(path.join(HERE, 'benchmark.html'), 'r') as f:
        return f.read().replace('{{ stacks }}', json.dumps(['spruce', 'pine']))


@app.route('/fetch/<stack_name>', methods=['GET'])
def get_stack_data(stack_name):
    s3 = boto3.client('s3')
    res = s3.list_objects_v2(Bucket=BUCKET, Prefix=f'{BASE}/{stack_name}/')
    print(res)
    return jsonify({})
