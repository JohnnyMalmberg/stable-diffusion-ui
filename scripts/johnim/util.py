def preprocess(user_input):
    return process_buffer(create_buffer(user_input))

def create_buffer(user_input):
    preproc_buffer = [('NOP', None)]
    target = preproc_buffer

    for i in range(len(user_input)):
        line = user_input[i]
        if line.startswith('$'):
            if line.startswith('$count'):
                split = line.split(' ')
                splitb = split[2].split('..')
                new_target = [('FOR', target, '$' + split[1], range(int(splitb[0]), int(splitb[1])))]
                target.append(new_target)
                target = new_target
            elif line.startswith('$for'):
                split = line.split(' ', 2)
                new_target = [('FOR', target, '$' + split[1], eval(split[2]))]
                target.append(new_target)
                target = new_target
            elif line.startswith('$$'):
                target = target[0][1]
        else:
            target.append(line)
    return preproc_buffer

def serialize_buffer(buf, pfx=''):
    out = []
    if buf[0][0] == 'NOP':
        for entry in buf[1:]:
            if type(entry) == str:
                out.append(pfx + entry)
            elif type(entry) == list:
                out.extend(serialize_buffer(entry, pfx + '  '))
            else:
                print('Bad preproc buffer construction')
    elif buf[0][0] == 'FOR':
        out.append(f'{pfx}for {buf[0][2]} in {buf[0][3]}:')
        for entry in buf[1:]:
            if type(entry) == str:
                out.append(pfx + entry)
            elif type(entry) == list:
                out.extend(serialize_buffer(entry, pfx + '  '))
            else:
                print('Bad preproc buffer construction')
    else:
        print(f'Unknown buffertype {buf[0][0]}')
    return out


def process_buffer(buf):
    out = []
    if buf[0][0] == 'NOP':
        for entry in buf[1:]:
            if type(entry) == str:
                out.append(entry)
            elif type(entry) == list:
                out.extend(process_buffer(entry))
            else:
                print(f'Bad preprocessor buffer construction.')
    elif buf[0][0] == 'FOR':
        for x in buf[0][3]:
            for entry in buf[1:]:
                if type(entry) == str:
                    out.append(entry.replace(buf[0][2], str(x)))
                elif type(entry) == list:
                    out.extend([e.replace(buf[0][2], str(x)) for e in process_buffer(entry)])
                else:
                    print('Bad preprocessor buffer construction.')
    else:
        print(f'Unknown buffertype {buf[0][0]}')
    return out
