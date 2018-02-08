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
        return commands[str(message['cmd'])](message['payload'])
    except KeyError as e:
        print("No valid callback found for function {} or no payload or error threw".format(message['cmd']))
        print(e)