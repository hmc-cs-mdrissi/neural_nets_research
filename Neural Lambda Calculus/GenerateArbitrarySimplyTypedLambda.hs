module GenerateArbitrarySimplyTypedLambda where

import Test.QuickCheck
import Data.Aeson hiding (Number)

import ArbitrarySimplyTypedLambda
import SimplyTypedLambdaParser
import SimplyTypedLambdaInterp

import System.Environment
import System.Exit
import Data.ByteString.Lazy (writeFile)
import Data.List (foldr, isPrefixOf)
import Control.Monad
import Prelude hiding (writeFile)
import qualified Data.Map as Map

-- 
-- GenerateArbitrarySimplyTypedLambda.hs
-- A set of functions for generating simply typed lambda calculus programs for output to a file.
-- 

data Config = Config {lcFileName :: String,
                      lcCount :: Int,
                      difficulty :: Difficulty,
                      termLength :: Int}

defaultConfig :: Config
defaultConfig = Config {lcFileName = "simplyTypedLambda.json", lcCount = 5, difficulty = Easy, termLength = 10}

parseArgumentsHelper :: Config -> String -> IO Config
parseArgumentsHelper cfg opt | "-lcFileName=" `isPrefixOf` opt = pure $ cfg {lcFileName = drop 12 opt}
                             | "-lcCount=" `isPrefixOf` opt = pure $ cfg {lcCount = read $ drop 9 opt}
                             | "-difficulty=" `isPrefixOf` opt = pure $ cfg {difficulty = read $ drop 12 opt}
                             | "-termLength=" `isPrefixOf` opt = pure $ cfg {termLength = read $ drop 12 opt}
                             | otherwise = die "You used an option that wasn't present."

parseArguments :: IO Config
parseArguments = do args <- getArgs
                    foldM parseArgumentsHelper defaultConfig args

generateArbitraryLC :: Difficulty -> Int -> Int -> IO [LambdaExpression]
generateArbitraryLC difficulty count exprLength = generate $ vectorOf count $ (arbitrarySizedSimplyTypedLambdaWithDifficulty difficulty)

makeApplication :: LambdaExpression -> Integer -> (Integer, Integer)
makeApplication lc_prog input = case evalCBV $ (BinaryOper lc_prog Application (Number input)) of
                                     Left err -> error (show err)
                                     Right expr -> let (Number output) = expr in (input, output)
                                    

makeInputOutputTuples :: LambdaExpression -> [(Integer, Integer)]
makeInputOutputTuples lc_prog = map (makeApplication lc_prog) [0 .. 10]

appendInterpreterOutput :: LambdaExpression -> (LambdaExpression, [(Integer, Integer)])
appendInterpreterOutput lc_prog = (lc_prog, makeInputOutputTuples lc_prog)

interpretAllProgs :: [LambdaExpression] -> [(LambdaExpression, [(Integer, Integer)])]
interpretAllProgs lc_progs = map appendInterpreterOutput lc_progs


main :: IO ()
main = do cfg <- parseArguments
          lc_progs <- generateArbitraryLC (difficulty cfg) (lcCount cfg) (termLength cfg)
          writeFile (show (difficulty cfg) ++ "-" ++ (lcFileName cfg)) $ encode (interpretAllProgs lc_progs)

