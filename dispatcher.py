commands = {}

def callback(command_name):
    def deco(func):
        if command_name not in commands:
            commands[command_name] = func
        else:
            print("Callback {} already declared".format(command_name))
    return deco

def call(message):
    try:
        commands[str(message['cmd'])](message['payload'])
    except KeyError:
        print("No valid callback found for function {} or no payload".format(message['cmd']))