from dataclasses import dataclass

@dataclass
class Processor:
    id: int
    power: int
    context_switched: bool
    previous_task_instance: int

    #returns true/false if tasks id compelte
    def process(self, task):

        if task.instance_id != self.previous_task_instance:
            self.context_switched = True
        else:
            self.context_switched = False

        task.exec_time -= self.power
        if task.exec_time <= 0:
            return True
        else:
            return False

    @classmethod
    def create_homogeneous_pset(cls, num_processors, power):
        pset = []
        for i in range(num_processors):
            pset.append(cls(i, power, 0, -1))

        return pset

    @classmethod
    def create_hetero_pset(cls, powers):
        pset = []
        for i,p in enumerate(powers):
            pset.append(cls(i,p, 0, -1))

        return pset 