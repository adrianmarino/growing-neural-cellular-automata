from lib.command.test_command import TestCommand
from lib.command.train_command import TrainCommand


class CommandFactory:
    @staticmethod
    def create(action):
        if 'test' == action:
            return TestCommand()
        elif 'train' == action:
            return TrainCommand()
        else:
            raise Exception('Unknown command!')
