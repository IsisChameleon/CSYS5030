#------------------------------------------------------------
# LOAD GAMES FROM TEXT FILES
#------------------------------------------------------------

import re
from pathlib import Path
import numpy as np
import pandas as pd
from week1 import entropyEmpirical

# moves = { 'scissors':0, 'paper':1, 'rock':2}
def getResult(game1: int, game2: int):
    results = [[0,1, -1], [-1, 0, 1], [1,-1,0]]
    result=results[game1][game2]
    return result

def loadGames(baseUriStr: str):

    player1Regex = re.compile('(?<=Player1:\s).*(?=;)')
    player2Regex = re.compile('(?<=Player2:\s).*(?=$)')
    game1Regex = re.compile('(?<=^)[012](?=\s)')
    game2Regex = re.compile('(?<=\s)[012](?=$)')

    baseUri=Path(baseUriStr)
    txtFiles=[x for x in baseUri.glob('**/*.txt')]

    # game = [game#, player1, player2, moveplayer1, moveplayer2]
    filenum=0
    games=list()
    for tf in txtFiles:
        with open(tf, 'rt') as f:
            lines = f.readlines()
    
        for i, line in enumerate(lines):
            if (i == 0):
                player1 = re.findall(player1Regex, line)[0].lower()
                player2 = re.findall(player2Regex, line)[0].lower()
            else:
                game1 =  int(re.findall(game1Regex, line)[0])
                game2 =  int(re.findall(game2Regex, line)[0])
                result = getResult(game1, game2)
                result2 = getResult(game2, game1)
                games.append([filenum, player1, player2, game1, game2, result, result2])

        filenum+=1

    return games

#------------------------------------------------------------
# LIST PLAYERS
#------------------------------------------------------------


def listPlayers(games: list):
    listPs = list()
    listPs += [ [game[1], game[2]] for game in games]
    listPs = np.array(listPs).flatten()
    Ps=list(set(listPs))
    return Ps


#------------------------------------------------------------
#   GET 1 PLAYER MOVES AND RESULTS ACCROSS ALL GAMES
#------------------------------------------------------------


def getPlayerAllMovesAndScores(playerName: str, games: np.array):
    playerName = playerName.lower()
    df_games = pd.DataFrame(games, columns = ['Game#', 'Player1', 'Player2', 'Move1', 'Move2', 'Result1', 'Result2' ])
    d1 = df_games.loc[ df_games.Player1 == playerName]
    moveWhen1 = list(d1.Move1)
    winWhen1 = list(d1.Result1)
    d2 = df_games.loc[ df_games.Player2 == playerName]
    moveWhen2 = list(d2.Move2)
    winWhen2 = list(d2.Result2)

    return moveWhen1 + moveWhen2, winWhen1 + winWhen2

def getPlayerAndOpponentAllMovesAndScores(playerName: str, games: np.array):
    playerName = playerName.lower()
    df_games = pd.DataFrame(games, columns = ['Game#', 'Player1', 'Player2', 'Move1', 'Move2', 'Result1', 'Result2' ])
    d1 = df_games.loc[ df_games.Player1 == playerName]
    moveWhen1 = list(d1.Move1)
    winWhen1 = list(d1.Result1)
    moveOpponentWhen1 = list(d1.Move2)
    d2 = df_games.loc[ df_games.Player2 == playerName]
    moveWhen2 = list(d2.Move2)
    winWhen2 = list(d2.Result2)
    moveOpponentWhen2 = list(d1.Move1)

    return moveWhen1 + moveWhen2, winWhen1 + winWhen2, moveOpponentWhen1 + moveOpponentWhen2

def getPlayerEntropyAndRatios(players: list, games: np.array):
    playersGameEntropy=dict()
    for player in players:
        playersGameEntropy[player]=None

    scores=list()
    entropies=list()
    lossRatios=list()
    winRatios=list()
    for player in players:
        moves, gains = getPlayerAllMovesAndScores(player, games)
        playersGameEntropy[player] = entropyEmpirical(moves)
        gains=np.array(gains)
        score=np.sum(gains)
        winRatio=len(list(gains[gains==1]))/len(list(gains))
        lossRatio=len(list(gains[gains==-1]))/len(list(gains))
        winRatios.append(winRatio)
        lossRatios.append(lossRatio)
        scores.append(score)
        entropies.append(playersGameEntropy[player])
        print('{} plays with entropy {} and in average a score of {}'.format(player, playersGameEntropy[player], score))
        print('winRatio {}, lossRatio {}'.format(winRatio, lossRatio))

    return entropies, winRatios, lossRatios
