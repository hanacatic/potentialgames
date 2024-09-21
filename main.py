from game import Game 

def mu(action_profile):
    return 1.0/8.0

def utility_function(player_action, opponents_actions):
    return 1

if __name__ == '__main__':
    
    action_profile = [1, 1]
    no_players = 2

    action_space = [1, 2, 3, 4]

    game = Game(no_players, action_space, [utility_function, utility_function], mu)
    
    print(game.action_profile)
    
    game.play()
    print(game.action_profile)
    
