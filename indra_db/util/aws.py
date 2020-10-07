import re
import boto3


def uncamel(word):
    return re.sub(r'([a-z])([A-Z])', '\g<1>_\g<2>', word).lower()


def get_role_kwargs(role):
    sts = boto3.client('sts')

    # Check the current role
    kwargs = {}
    ident = sts.get_caller_identity()
    if role and not ident['Arn'].endswith(role):
        # If the role is not the default, assume that role.
        new_role_arn = "arn:aws:iam::%s:role/%s" % (ident['Account'], role)
        res = sts.assume_role(RoleArn=new_role_arn,
                              RoleSessionName="AssumeRoleReadonlyDBUpdate")
        kwargs = {'aws_' + uncamel(k): v for k, v in res['Credentials'].items()
                  if 'expiration' not in k.lower()}

    return kwargs, ident
