import boto3
from indra_db.util.constructors import get_primary_db


def lambda_handler(event, context):
    print('Event:', event)
    print('Context:', context)

    username = event['userName']
    email = event['request']['userAttributes']['email']

    db = get_primary_db()
    key = db._get_api_key(username)
    if key is not None:
        return {'status': 400, 'body': 'User already exists.'}
    dbid, api_key = db._add_auth(username, email)
    ses = boto3.client('ses')
    ses.send_email(Source='noreply@indra.bio',
                   Destination={'ToAddresses': [email]},
                   Message={'Subject': {'Data': 'Your API key',
                                        'Charset': 'ascii'},
                            'Body': {'Text': {'Data': api_key,
                                              'Charset': 'ascii'}}})
    return event

