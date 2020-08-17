class Relation:
    def __init__(self, agent_dict, stmt_type=None, hashes=None):
        self.agent_dict = agent_dict
        self.stmt_type = stmt_type
        self.hashes = hashes

    def expand(self, ro):
        if self.hashes is not None:
            ro.select_all()
        elif self.stmt_type is not None:
            pass
        else:
            pass
