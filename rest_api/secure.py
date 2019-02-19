import boto3


def get_gateway_client(role='SUDO'):
    """Get a boto3 client to the gateway with SUDO role permissions."""
    sts = boto3.client('sts')
    resp = sts.get_caller_identity()
    acct_id = resp['Account']
    aro = sts.assume_role(RoleArn='arn:aws:iam::%s:role/%s' % (acct_id, role),
                          RoleSessionName='Assuming%s' % role)
    creds = aro['Credentials']
    agc = boto3.client('apigateway', aws_access_key_id=creds['AccessKeyId'],
                       aws_secret_access_key=creds['SecretAccessKey'],
                       aws_session_token=creds['SessionToken'])
    return agc


def add_authorizers():
    """Add authorizers to the api endpoint."""
