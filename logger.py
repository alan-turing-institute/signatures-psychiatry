import datetime
import os

class Logger:
    name = None

    def __init__(self, name):
        '''Initialises the logger.

        Arguments:
                name (str): Name of log file.

        '''

        self.name = name
        self.path = "log/%s.log"%self.name

        timestamp = datetime.datetime.now()
        header = "================ %s ================"%timestamp
        self.footer = "="*len(header) + "\n\n"
        s = "%s\n%s"%(header, self.footer)

        try:
            with open(self.path, "a") as handle:
                handle.write(s)

        except:
            print("Error: Couldn't write to path %s"%self.path)

    def log(self, s, verbose=True):
        '''Initialises the logger.

        Arguments:
                s (str): Text to be logged.
                verbose (bool): If True, prints the text.

        '''

        if verbose:
            print s

        try:
            lines = open(self.path, 'r').readlines()

            # now edit the last line of the list of lines
            lines[-2] = s + "\n"
            lines[-1] = self.footer

            # now write the modified list back out to the file
            open(self.path, 'w').writelines(lines)
        except:
            print("Error: Couldn't write to path %s"%self.path)
