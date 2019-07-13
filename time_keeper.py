from timeit import default_timer as timer#timer 
import numpy as np

total_game_time = 0
act_time = 0
move_to_leaf_time = 0
evaluate_leaf_time = 0
get_preds_time = 0
backfill_time = 0
add_time = 0

take_action_time = 0

predict_time = 0

times = np.zeros(10)

def print_ratios(total_game_time, move_to_leaf_time, evaluate_leaf_time, get_preds_time, backfill_time, take_action_time, predict_time):
    #act_ratio = act_time/ total_game_time
    move_to_leaf_ratio = 100 * move_to_leaf_time/ total_game_time
    evaluate_leaf_ratio = 100 * evaluate_leaf_time/ total_game_time
    get_preds_ratio =  100 * get_preds_time/ evaluate_leaf_time
    backfill_ratio = 100 * backfill_time / total_game_time
    other = 100 - move_to_leaf_ratio - evaluate_leaf_ratio - backfill_ratio

    take_action_ratio = 100 * take_action_time/total_game_time

    predict_ratio = 100 * predict_time/total_game_time

    print("Total Game Time: {0}".format(total_game_time))
    print("move_to_leaf_ratio: {0}%".format(move_to_leaf_ratio))
    print("evaluate_leaf_ratio: {0}%".format(evaluate_leaf_ratio))
    print("backfill_ratio: {0}".format(backfill_ratio))
    #print("get_preds_ratio: {0}%".format(get_preds_ratio))
    print("other: {0}%".format(move_to_leaf_ratio))

    print("eval_time: {0} get_preds_time: {1}".format(evaluate_leaf_time, get_preds_time))
    print("% of eval_leaf taken up by get_preds: {0}%".format(get_preds_ratio))
   
    print("take_action_time: {0} take_action_ratio: {1}".format(take_action_time, take_action_ratio))

    print("\n\nmodel.predict_ratio: {0}%".format(predict_ratio))
    move_to_leaf_time = 0
    evaluate_leaf_time = 0
    get_preds_time = 0
    backfill_time = 0

def diagnose_time(times):
    ratios = []
    total_time = np.sum(times)
    for i, time in enumerate(times):
        ratios.append(100 * time / total_time)
    
    for i, ratio in enumerate(ratios):
        print("section {0}: {1}%".format(i,ratio))
    



"""class time_keeper():
    def __init__(self):
        self.total_game_time = 0
        self.act_time = 0
        self.move_to_leaf_time = 0
        self.evaluate_leaf_time = 0
        self.get_preds_time = 0
        self.backfill_time = 0

    def print_ratios():
        #act_ratio = act_time/ total_game_time
        move_to_leaf_ratio = 100 * self.move_to_leaf_time/ self.total_game_time
        evaluate_leaf_ratio = 100 * self.evaluate_leaf_time/ self.total_game_time
        #get_preds_ratio =  100 * get_preds_time/ total_game_time
        backfill_ratio = 100 * self.backfill_time / self.total_game_time
        other = 100 - move_to_leaf_ratio - evaluate_leaf_ratio - get_preds_ratio

        print("Total Game Time: {0}".format(self.total_game_time))
        print("move_to_leaf_ratio: {0}%".format(move_to_leaf_ratio))
        print("evaluate_leaf_ratio: {0}%".format(evaluate_leaf_ratio))
        print("backfill_ratio: {0}".format(backfill_ratio))
        #print("get_preds_ratio: {0}%".format(get_preds_ratio))
        print("other: {0}%".format(move_to_leaf_ratio))

        self.move_to_leaf_time = 0
        self.evaluate_leaf_time = 0
        self.get_preds_time = 0
        self.backfill_time = 0"""