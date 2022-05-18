def get_real_flag(data):
    content = data[6:].replace('a', '4')
    content = content.replace('s', '5')
    content = content.replace('i', '1')
    content = content.replace('e', '3')
    content = content.replace(' ', '_')
    content = content.replace('o', '0')
    content = content.replace('t', '7')
    return data[:5].upper() + '{' + content + '}'
