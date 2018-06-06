module GenerateArbitraryForTests where

import Test.QuickCheck
import Data.Aeson

import ArbitraryForTests
import LambdaParser
import ForParser
import ForInterp

import System.Environment
import System.Exit
import Data.ByteString.Lazy (writeFile)
import Data.List (isPrefixOf)
import Data.Map (Map, toList, lookup)
import qualified Data.Map as Map
import Control.Monad
import Prelude hiding (writeFile)

--
-- GenerateArbitraryTests.hs
-- A set of functions for generating a set of programs and storing their JSON encodings
-- in a file.
--

-- Typical term lengths: Easy = 5, Medium = 8, Hard = 10, VeryHard = 15

data Config = Config {forFileName :: String,
                      forCount :: Int,
                      difficulty :: Difficulty,
                      termLength :: Int}

defaultConfig :: Config
defaultConfig = Config {forFileName = "training_For.json", forCount = 100000, difficulty = VeryHard, termLength = 15}

parseArgumentsHelper :: Config -> String -> IO Config
parseArgumentsHelper cfg opt | "-forFileName=" `isPrefixOf` opt = pure $ cfg {forFileName = drop 13 opt}
                             | "-forCount=" `isPrefixOf` opt = pure $ cfg {forCount = read $ drop 10 opt}
                             | "-difficulty=" `isPrefixOf` opt = pure $ cfg {difficulty = read $ drop 12 opt}
                             | "-termLength=" `isPrefixOf` opt = pure $ cfg {termLength = read $ drop 12 opt}
                             | otherwise = die "You used an option that wasn't present."

parseArguments :: IO Config
parseArguments = do args <- getArgs
                    foldM parseArgumentsHelper defaultConfig args

makeContext :: Int -> Context
makeContext initialValue = Map.fromList [("a0", initialValue)]

getA0ValueFromContext :: Context -> Int
getA0ValueFromContext context = case (Map.lookup "a0" context) of
                                    Just newValue -> newValue
                                    _ -> error "Error: Lookup from new context failed."

makeApplication :: ProgFor -> Int -> (Int, Int)
makeApplication for_prog input = let currentContext = makeContext input in
                                     case (eval currentContext for_prog) of
                                        Just newContext -> (input, (getA0ValueFromContext newContext))
                                        _ -> error "Error: For program evaluation failed."

makeInputOutputTuples :: ProgFor -> [(Int, Int)]
makeInputOutputTuples for_prog = map (makeApplication for_prog) [0 .. 10]

appendInterpreterOutput :: ProgFor -> (ProgFor, [(Int, Int)])
appendInterpreterOutput for_prog = (for_prog, makeInputOutputTuples for_prog)

interpretAllProgs :: [ProgFor] -> [(ProgFor, [(Int, Int)])]
interpretAllProgs for_progs = map appendInterpreterOutput for_progs

generateArbitraryFor :: Difficulty -> Int -> Int -> IO [ProgFor]
generateArbitraryFor difficulty count exprLength = generate $ vectorOf count $ fst <$> (arbitrarySizedProgForWithDifficulty difficulty 0 exprLength)

showForProg :: ProgFor -> IO ()
showForProg for_prog = let pairLC = appendInterpreterOutput for_prog in
                       let for_expr = fst pairLC in
                       let outputList = snd pairLC in
                       putStrLn ("PROGRAM: " ++ (show for_expr) ++ ("\nOUTPUT: ") ++ (show outputList))

main :: IO ()
main = do cfg <- parseArguments
          for_progs <- generateArbitraryFor (difficulty cfg) (forCount cfg) (termLength cfg)
          writeFile (forFileName cfg) $ encode for_progs

