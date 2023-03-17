import gc, traceback
import torch, clip
import torch.optim as optim
from tqdm import tqdm
from torch import autocast, rand
from transformers import CLIPTextModel, CLIPTokenizer
from random import randint, choice, uniform

class Clipper(object):
    def __init__(self, state):
        self.steps = 0
        self.rate = 0.0001
        self.state = state
        self.state.embeds = {}
        self.state.tokens = {}
        self.model = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14',).to('cuda')
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

    # c_<command name>
    # all commands take a single string as their argument

    def do_ucommand(self, command, args):
        return command != "exit"

    def do_command(self, command, args):
        print(f'[{command}] [{args}]')
        if command == 'exit':
            return False
        try:
            Clipper.__dict__[f'c_{command}'](self, args)
        except Exception as e:
            print(f'{"="*10}\n\n{traceback.format_exc()}\n{"="*10}')
        
        gc.collect()
        torch.cuda.empty_cache()
        return True

    # lmao
    #def c_eval(self, args):
    #    eval(args)

    def c_test(self, foo):
        print(foo)

    def c_steps(self, steps_str):
        self.steps = int(steps_str)

    def c_rate(self, rate_str):
        self.rate = float(rate_str)

    def c_load(self, path):
        self.state.embeds[path] = torch.load(f'clip_embeds/{path}.pt').to('cuda')

    def c_save(self, path):
        torch.save(self.state.embeds[path], f'clip_embeds/{path}.pt')

    def c_refresh(self, no_arg):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14',).to('cuda')

    def c_embed(self, args):
        arg_split = args.split(' ', 1)
        embed_path = arg_split[0]
        if len(arg_split) > 1:
            prompt = arg_split[1]
        else:
            prompt = ""
        tokens = self.tokenizer(prompt, 
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding='max_length',
            return_tensors='pt'
        )['input_ids'].to('cuda')
        self.state.tokens[embed_path] = tokens
        #self.state.embeds[embed_path] = self.model.get_text_features(tokens)
        self.state.embeds[embed_path] = self.model(input_ids=tokens).last_hidden_state

    def c_random(self, args):
        arg_split = args.split(' ')
        embed_path = arg_split[0]
        r = rand((1,77,768)).to('cuda')
        self.state.embeds[embed_path] = (r - 0.5) * 2.0

    def c_random_gen(self, args):
        arg_split = args.split(' ')
        embed_path = arg_split[0]
        r = rand((10,77,768)).to('cuda')
        self.state.embeds[embed_path] = (r - 0.5) * 2.0

    def c_dump(self, no_arg):
        print(self.state.embeds)
        print(self.state.tokens)

    def c_dump_shape(self, no_arg):
        for key in self.state.embeds:
            print(f'{key}: {self.state.embeds[key].shape}')
        for key in self.state.tokens:
            print(f'{key}: {self.state.tokens[key].shape}')

    def splice_embeds(self, embed1, embed2):
        vecs = [embed1[0][i] if randint(0,1) else embed2[0][i] for i in range(77)]
        return torch.stack(vecs)
        
    def c_splice(self, args):
        arg_split = args.split(' ')
        path1 = arg_split[0]
        path2 = arg_split[1]
        pathout = arg_split[2]
        embed1 = self.state.embeds[path1]
        embed2 = self.state.embeds[path2]
        embedout = self.splice_embeds(embed1, embed2)[None]
        self.state.embeds[pathout] = embedout

    def c_breed(self, args):
        arg_split = args.split(' ')
        path1 = arg_split[0]
        if '@' in path1:
            s = path1.split('@')
            path1 = s[1]
            n1 = int(s[0])
            embed1 = self.state.embeds[path1][n1][None]
        else:
            embed1 = self.state.embeds[path1]
        path2 = arg_split[1]
        if '@' in path2:
            s = path2.split('@')
            path2 = s[1]
            n2 = int(s[0])
            embed2 = self.state.embeds[path2][n2][None]
        else:
            embed2 = self.state.embeds[path2]
        pathout = arg_split[2]
        splices = [self.splice_embeds(embed1, embed2) for i in range(10)]
        # Add noise to genome
        splices = [splice + (torch.rand_like(splice) - 0.5) * choice([0.1, 0, 0, 0, 0, 0.05, 0.075, 0.025]) for splice in splices]
        for splice in splices:
            # Swap genes
            for x in range(choice([0, 0, 0,1,1,2,4,8])):
                a = randint(0, 76)
                b = randint(0, 76)
                while b == a:
                    b = randint(0, 76)
                temp = splice[a]
                splice[a] = splice[b]
                splice[b] = temp
            # Duplicate genes
            for x in range(choice([0, 0, 0, 0, 1, 1, 2, 3])):
                a = randint(0, 76)
                b = randint(0, 76)
                while b == a:
                    b = randint(0, 76)
                splice[a] = splice[b]
            # Invert genes
            for x in range(choice([0, 0, 0, 0, 1, 1, 2, 3])):
                a = randint(0, 76)
                splice[a] = -splice[a]
            # Individual value mutation
            for x in range(choice([0, 1, 1, 1, 2, 2, 4, 8])):
                a = randint(0, 76)
                b = randint(0, 767)
                splice[a][b] = uniform(-1, 1)

        self.state.embeds[pathout] = torch.stack(splices)

    def c_breed_mut(self, args):
        arg_split = args.split(' ')
        path1 = arg_split[0]
        if '@' in path1:
            s = path1.split('@')
            path1 = s[1]
            n1 = int(s[0])
            embed1 = self.state.embeds[path1][n1][None]
        else:
            embed1 = self.state.embeds[path1]
        path2 = arg_split[1]
        if '@' in path2:
            s = path2.split('@')
            path2 = s[1]
            n2 = int(s[0])
            embed2 = self.state.embeds[path2][n2][None]
        else:
            embed2 = self.state.embeds[path2]
        pathout = arg_split[2]
        splices = [self.splice_embeds(embed1, embed2) for i in range(10)]
        splices = [splice + (torch.rand_like(splice) - 0.5) * choice([0.1, 0, 0, 0, 0, 0.05, 0.075, 0.025]) for splice in splices]
        self.state.embeds[pathout] = torch.stack(splices)

    def c_breed_premut(self, args):
        arg_split = args.split(' ')
        path1 = arg_split[0]
        if '@' in path1:
            s = path1.split('@')
            path1 = s[1]
            n1 = int(s[0])
            embed1 = self.state.embeds[path1][n1][None]
        else:
            embed1 = self.state.embeds[path1]
        path2 = arg_split[1]
        if '@' in path2:
            s = path2.split('@')
            path2 = s[1]
            n2 = int(s[0])
            embed2 = self.state.embeds[path2][n2][None]
        else:
            embed2 = self.state.embeds[path2]
        pathout = arg_split[2]
        splices = [self.splice_embeds(embed1, embed2) for i in range(10)]
        self.state.embeds[pathout] = torch.stack(splices)

    def c_clear(self, no_arg):
        del self.state.embeds, self.state.tokens
        gc.collect()
        torch.cuda.empty_cache()
        self.state.embeds = {}
        self.state.tokens = {}

    def c_learn_x(self, args):
        arg_split = args.split(' ')
        path_1 = arg_split[0]
        path_2 = arg_split[1]
        path_out = arg_split[2]

        tokens = self.state.tokens[path_1].detach().clone()

        token_embed = self.model.get_text_features(tokens)
        fixed_embed = self.state.embeds[path_2].detach().clone()

        token_embed /= token_embed.norm(dim=-1, keepdim=True)
        fixed_embed /= fixed_embed.norm(dim=-1, keepdim=True)

        fixed_embed_T = fixed_embed.T

        sim = token_embed @ fixed_embed_T

        loss = -sim

        optimizer = optim.Adam([tokens], lr=self.rate)

        print(tokens)

        for i in tqdm(range(self.steps)):
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            token_embed = self.model.get_text_features(tokens)
            token_embed /= token_embed.norm(dim=-1, keepdim=True)
            sim = token_embed @ fixed_embed_T
            loss = -sim

        print(tokens)

        self.state.embeds[path_out] = self.model.text_model(input_ids=tokens).last_hidden_state.detach().clone()

    def c_learn(self, args):
        arg_split = args.split(' ')
        path_1 = arg_split[0]
        path_2 = arg_split[1]
        path_out = arg_split[2]

        tokens = self.state.tokens[path_1].detach().clone()

        token_embed = self.model.get_text_features(tokens)
        fixed_embed = self.state.embeds[path_2].detach().clone()

        token_embed /= token_embed.norm(dim=-1, keepdim=True)
        fixed_embed /= fixed_embed.norm(dim=-1, keepdim=True)

        fixed_embed_T = fixed_embed.T

        sim = token_embed @ fixed_embed_T

        loss = -sim

        optimizer = optim.Adam(self.model.text_model.parameters(), lr=self.rate)


        for i in tqdm(range(self.steps)):
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            token_embed = self.model.get_text_features(tokens)
            token_embed /= token_embed.norm(dim=-1, keepdim=True)
            sim = token_embed @ fixed_embed_T
            loss = -sim

        self.state.embeds[path_out] = self.model.text_model(input_ids=tokens).last_hidden_state.detach().clone()

def clipper_thread(state):
    with autocast('cuda'):
        clipper = Clipper(state)
        should_continue = True

        print("Clipper Thread Launched")

        while not should_continue:
            if not state.urgent_queue.empty():
                (ucommand, uargs) = state.urgent_queue.get()
                should_continue = do_ucommand(ucommand, uargs, state)
            (command, args, cb) = state.clipper_queue.get()
            try:
                should_continue = clipper.do_command(command, args)
            except Exception as ex:
                print(f'[Exception | Clipper | {ex}] {traceback.format_exc()}')

def main():
    with autocast('cuda'):
        clipper = Clipper()
        should_exit = False

        print("Welcome to Promptcrafter")

        while not should_exit:
            user_input = input('> ')
            if user_input == 'exit':
                should_exit = True
                continue
            split_input = user_input.split(' ', 1)
            command = split_input[0]
            if len(split_input) > 1:
                args = split_input[1]
            else:
                args = ''
            clipper.do_command(command, args)

    print('Goodbye.')




if __name__ == '__main__':
    main()