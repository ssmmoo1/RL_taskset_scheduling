from dataclasses import dataclass

@dataclass
class Processor:
    id: int
    power: int

    #returns true/false if tasks id compelte
    def process(self, task):
        task.exec_time -= self.power
        if task.exec_time <= 0:
            return True
        else:
            return False

    @classmethod
    def create_homogeneous_pset(cls, num_processors, power):
        pset = []
        for i in range(num_processors):
            pset.append(cls(i, power))

        return pset

    @classmethod
    def create_hetero_pset(cls, powers):
        pset = []
        for i,p in enumerate(powers):
            pset.append(cls(i,p))

        return pset 