from utils import *
from main import *
import chess.engine
STOCKFISH_PATH = '/path/to/your/stockfish.exe'

class TestDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, all_moves_dict, elo_dict):
        
        self.all_moves_dict = all_moves_dict
        self.data = self._expand_position(data.values.tolist())
        self.elo_dict = elo_dict

    def _expand_position(self, data):
        
        ret = []
        for line in data:
            for elo in range(1100, 2000, 100):
                ret.append([line[0], line[1], elo, elo])
        
        return ret

    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        fen, move, elo_self, elo_oppo = self.data[idx]

        if fen.split(' ')[1] == 'w':
            board = chess.Board(fen)
        elif fen.split(' ')[1] == 'b':
            board = chess.Board(fen).mirror()
            move = mirror_move(move)
        else:
            raise ValueError(f"Invalid fen: {fen}")
            
        board_input = board_to_tensor(board)
        
        elo_self = map_to_category(elo_self, self.elo_dict)
        elo_oppo = map_to_category(elo_oppo, self.elo_dict)
        
        legal_moves, _ = get_side_info(board, move, self.all_moves_dict)
        
        return fen, board_input, elo_self, elo_oppo, legal_moves


class TestDataset_mono(torch.utils.data.Dataset):
    
    def __init__(self, data, all_moves_dict, elo_dict):
        
        self.all_moves_dict = all_moves_dict
        self.data = data.values.tolist()
        self.elo_dict = elo_dict

    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        line = self.data[idx]
        fen, move, elo_self, elo_oppo = line[0], line[1], line[2], line[3]
        best_move = line[6]

        if fen.split(' ')[1] == 'w':
            board = chess.Board(fen)
        elif fen.split(' ')[1] == 'b':
            board = chess.Board(fen).mirror()
            move = mirror_move(move)
            best_move = mirror_move(best_move)
        else:
            raise ValueError(f"Invalid fen: {fen}")
            
        board_input = board_to_tensor(board)
        
        elo_self = map_to_category(elo_self, self.elo_dict)
        elo_oppo = map_to_category(elo_oppo, self.elo_dict)
        
        legal_moves, _ = get_side_info(board, move, self.all_moves_dict)
        
        return fen, board_input, elo_self, elo_oppo, legal_moves, best_move


def get_preds(model, dataloader, all_moves_dict_reversed, cfg_inference):
    
    # all_probs = []
    predicted_move_probs = []
    predicted_moves = []
    # predicted_win_probs = []
    
    model.eval()
    with torch.no_grad():
        
        for fens, boards, elos_self, elos_oppo, legal_moves in dataloader:
            
            if cfg_inference.gpu:
                boards = boards.cuda()
                elos_self = elos_self.cuda()
                elos_oppo = elos_oppo.cuda()
                legal_moves = legal_moves.cuda()

            logits_maia, _, logits_value = model(boards, elos_self, elos_oppo)
            logits_maia_legal = logits_maia * legal_moves
            probs = logits_maia_legal.softmax(dim=-1)

            # all_probs.append(probs.cpu())
            predicted_move_probs.append(probs.max(dim=-1).values.cpu())
            predicted_move_indices = probs.argmax(dim=-1)
            for i in range(len(fens)):
                fen = fens[i]
                predicted_move = all_moves_dict_reversed[predicted_move_indices[i].item()]
                if fen.split(' ')[1] == 'b':
                    predicted_move = mirror_move(predicted_move)
                predicted_moves.append(predicted_move)

            # predicted_win_probs.append((logits_value / 2 + 0.5).cpu())
    
    # all_probs = torch.cat(all_probs).cpu().numpy()
    predicted_move_probs = torch.cat(predicted_move_probs).numpy()
    # predicted_win_probs = torch.cat(predicted_win_probs).numpy()
    
    return predicted_move_probs, predicted_moves


def get_preds_mono(model, dataloader, all_moves_dict, cfg_inference):
    
    # all_probs = []
    # predicted_move_probs = []
    # predicted_moves = []
    predicted_win_probs = []
    best_move_probs = []
    
    model.eval()
    with torch.no_grad():
        
        for fens, boards, elos_self, elos_oppo, legal_moves, best_move in dataloader:
            
            if cfg_inference.gpu:
                boards = boards.cuda()
                elos_self = elos_self.cuda()
                elos_oppo = elos_oppo.cuda()
                legal_moves = legal_moves.cuda()

            logits_maia, _, logits_value = model(boards, elos_self, elos_oppo)
            logits_maia_legal = logits_maia * legal_moves
            probs = logits_maia_legal.softmax(dim=-1)

            # all_probs.append(probs.cpu())
            # predicted_move_probs.append(probs.max(dim=-1).values.cpu())
            # predicted_move_indices = probs.argmax(dim=-1)
            for i in range(len(fens)):
                best_move_probs.append(probs[i][all_moves_dict[best_move[i]]].item())
            #     fen = fens[i]
            #     predicted_move = all_moves_dict_reversed[predicted_move_indices[i].item()]
                predicted_win_probs.append((logits_value[i] / 2 + 0.5).item())
            #         predicted_move = mirror_move(predicted_move)
            #     predicted_moves.append(predicted_move)

            
    
    # all_probs = torch.cat(all_probs).cpu().numpy()
    # predicted_move_probs = torch.cat(predicted_move_probs).numpy()
    predicted_win_probs = torch.tensor(predicted_win_probs).numpy()
    best_move_probs = torch.tensor(best_move_probs).numpy()
    
    return best_move_probs, predicted_win_probs


def inference_batch(data):
    
    cfg = parse_args()
    cfg_inference = parse_inference_args()
    if cfg_inference.verbose:
        show_cfg(cfg_inference)

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    model = nn.DataParallel(model)
    
    checkpoint = torch.load(cfg_inference.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.module
    
    if cfg_inference.gpu:
        model = model.cuda()
    
    all_moves_dict_reversed = {v: k for k, v in all_moves_dict.items()}
    dataset = TestDataset(data, all_moves_dict, elo_dict)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=cfg_inference.batch_size, 
                                            shuffle=False, 
                                            drop_last=False,
                                            num_workers=cfg_inference.num_workers)
    if cfg_inference.verbose:
        dataloader = tqdm.tqdm(dataloader)
    predicted_move_probs, predicted_moves = get_preds(model, dataloader, all_moves_dict_reversed, cfg_inference)
    
    ret = pd.DataFrame(dataset.data, columns=['board', 'move', 'active_elo', 'opponent_elo'])
    ret['predicted_move'] = predicted_moves
    ret['predicted_move_prob'] = predicted_move_probs
    # data['predicted_win_prob'] = predicted_win_probs
    # data['all_probs'] = all_probs.tolist()
    
    return ret

def inference_batch_mono(data):
    
    cfg = parse_args()
    cfg_inference = parse_inference_args()
    if cfg_inference.verbose:
        show_cfg(cfg_inference)

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    model = nn.DataParallel(model)
    
    checkpoint = torch.load(cfg_inference.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.module
    
    if cfg_inference.gpu:
        model = model.cuda()
    
    # all_moves_dict_reversed = {v: k for k, v in all_moves_dict.items()}
    dataset = TestDataset_mono(data, all_moves_dict, elo_dict)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=cfg_inference.batch_size, 
                                            shuffle=False, 
                                            drop_last=False,
                                            num_workers=cfg_inference.num_workers)
    if cfg_inference.verbose:
        dataloader = tqdm.tqdm(dataloader)
    best_move_probs, predicted_win_probs = get_preds_mono(model, dataloader, all_moves_dict, cfg_inference)
    
    # ret = pd.DataFrame(dataset.data, columns=['board', 'move', 'active_elo', 'opponent_elo'])
    # ret['predicted_move'] = predicted_moves
    # ret['predicted_move_prob'] = predicted_move_probs
    # data['predicted_win_prob'] = predicted_win_probs
    # data['all_probs'] = all_probs.tolist()
    data['best_move_prob'] = best_move_probs
    data['predicted_win_prob'] = predicted_win_probs
    
    return data

def parse_inference_args(args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    # parser.add_argument('--batch_size', default=8192, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--model_path', default='../tmp/0.0001_8192_1e-05_MAIA2/epoch_2_2023-11.pgn.pt', type=str)
    # parser.add_argument('--model_path', default='../tmp/0.0001_8192_1e-05_MAIA2_s/epoch_1_2020-03.pgn.pt', type=str)
    parser.add_argument('--gpu', default=True, type=bool)

    return parser.parse_args(args)

def show_cfg(cfg):
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)

def filter_same_predicted_move(group):
    if group['predicted_move'].nunique() == 1:
        return None  # If all predicted moves are the same, return None to filter out
    return group  # Otherwise, return the group as is

def filter_positions(df):
    # Group the DataFrame by each set of 9 rows and apply the filtering function
    ret = pd.concat([filter_same_predicted_move(group) for _, group in df.groupby(df.index // 9)])
    return ret

def stockfish_eval(engine, line, limit):

    board = chess.Board(line[0])
    my_move = chess.Move.from_uci(line[4])
    
    info = engine.analyse(board, limit)

    pv = ', '.join([move.uci() for move in info['pv']])
    score = str(info['score'])[9: -1]

    best_move = info.get("pv")[0]
    best_move_score = info.get("score").white().score(mate_score=100000)  # Use a large mate score to handle checkmate scores

    # Make your move (replace 'your_move_uci' with the move you're analyzing, in UCI format)
    # your_move = chess.Move.from_uci('e2e4')  # Example move
    board.push(my_move)

    # Evaluate the position after your move
    info_after_your_move = engine.analyse(board, limit)
    your_move_score = info_after_your_move.get("score").white().score(mate_score=100000)

    # Adjust scores based on the player's perspective
    if board.turn == chess.BLACK:  # After making the move, it's the opponent's turn
        # For Black, invert the scores because Stockfish scores are from White's perspective
        best_move_score = -best_move_score
        your_move_score = -your_move_score

    # Calculate CP Loss
    cp_loss = best_move_score - your_move_score

    return best_move, cp_loss, pv, score

def transitional(df):
    
    df['transitional'] = False

    def process_group(group):
        # Determine the best move for the group as the most frequently occurring predicted_move
        best_move_counts = group['best_move'].value_counts()
        if len(best_move_counts) != 1:
            return
        best_move = best_move_counts.index[0]
        
        # Track if the group has switched to the best move and stayed there
        has_changed_to_best = False
        has_deviated_after_best = False
        
        for move in group['predicted_move']:
            if move == best_move:
                has_changed_to_best = True
            if has_changed_to_best and move != best_move:
                # If we have changed to the best move but then deviate to another, flag this
                has_deviated_after_best = True
                break  # No need to check further, the group fails the condition
        
        # If we've changed to the best move and not deviated after that, flag the group
        if has_changed_to_best and not has_deviated_after_best:
            group['transitional'] = True
            # print(group)
        
        return group

    # Apply the function to each group of 9 rows
    df = df.groupby(df.index // 9).apply(process_group).reset_index(drop=True)
    
    return df


def monotonic(df):
    
    def check_monotonic_increase(group):
        # Check if the best_move_prob values are monotonically increasing
        if group['best_move_prob'].is_monotonic_increasing:
            group['monotonic'] = True
        else:
            group['monotonic'] = False
        return group

    # Apply the function to each group of 9 rows
    df = df.groupby(df.index // 9).apply(check_monotonic_increase).reset_index(drop=True)
    return df


def fen_to_lichess_url(fen):
    base_url = "https://lichess.org/analysis/standard/"
    board_position = '_'.join(fen.split(" "))
    lichess_url = base_url + board_position
    return lichess_url


if __name__ == '__main__':
    
    # STOCKFISH_PATH = '/datadrive/josephtang/MAIA2/stockfish/src/stockfish'
    # engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    # engine.configure({"Threads": 16})
    # limit = chess.engine.Limit(depth=20)
    
    # data = pd.read_csv('../data/all_reduced_rapid.csv')
    # data = data[data.move_ply > 10][['board', 'move', 'active_elo', 'opponent_elo']].head(1000)
    # results = inference_batch(data)
    # filtered_results = filter_positions(results)

    # # add columns to results to include best_move and cp_loss
    # best_moves = []
    # cp_losses = []
    # pvs = []
    # scores = []
    # for line in tqdm.tqdm(filtered_results.values):
    #     best_move, cp_loss, pv, score = stockfish_eval(engine, line, limit)
    #     best_moves.append(best_move)
    #     cp_losses.append(cp_loss)
    #     pvs.append(pv)
    #     scores.append(score)
    # filtered_results['best_move'] = best_moves
    # filtered_results['cp_loss'] = cp_losses
    # filtered_results['pv'] = pvs
    # filtered_results['score'] = scores
    
    # filtered_results.to_csv('./filtered_results.csv', index=False)
    filtered_results = pd.read_csv('./filtered_results.csv')

    transitional_results = transitional(filtered_results)
    best_move_probs_added = inference_batch_mono(transitional_results)
    monotonic_results = monotonic(best_move_probs_added)
    monotonic_results['position_index'] = monotonic_results.index // 9
    monotonic_results['board'] = monotonic_results['board'].apply(fen_to_lichess_url)
    monotonic_results = monotonic_results[['position_index', 'board', 'move', 'score', 'active_elo', 'opponent_elo', 'predicted_move', 'predicted_move_prob', 'best_move', 'best_move_prob', 'predicted_win_prob', 'transitional', 'monotonic', 'pv']]
    
    monotonic_results.to_csv('./TransMono_results.csv', index=False)
    
    # data_novice = data[data['rounded_elo'] <= 1500][['board', 'move', 'active_elo', 'opponent_elo']]
    # data_intermediate = data[(data['rounded_elo'] > 1500) & (data['rounded_elo'] < 2000)][['board', 'move', 'active_elo', 'opponent_elo']]
    # data_advanced = data[data['rounded_elo'] >= 2000][['board', 'move', 'active_elo', 'opponent_elo']]
    # print(f'lens: {len(data_novice)}, {len(data_intermediate)}, {len(data_advanced)}', flush=True)
    
    # results = []
    # for split in [data_novice, data_intermediate, data_advanced]:
    #     inference_batch(split)
    #     print(round(len(split[split['predicted_move'] == split['move']]) / len(split), 4))
    
    # data_speed_test = pd.concat([data_novice] * 100, ignore_index=True)
    # inference_batch(data_speed_test)