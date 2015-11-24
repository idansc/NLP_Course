from optimizer import Optimizer


def get_parsed_data(filepath):
    with open(filepath, 'r') as f:
        word_tag_map = [("*","*")]
        for line in f:
#           we add start symbol
            for wordTag in line.split(' '):
                    word, tag = wordTag.split('_')
                    word_tag_map.append((word,tag))
            word_tag_map.append(("*","*"))
    return word_tag_map;


if __name__ == '__main__':
    print("Hello World!")
    
    y = Optimizer()
    y.optimize()

    print(get_parsed_data("../resources/train.wtag"))




