module GenerateArbitraryTests where

import Test.QuickCheck
import Data.Aeson

import ArbitraryTests
import LambdaParser
import ForParser

import System.Environment
import System.Exit
import Data.ByteString.Lazy (writeFile)
import Data.List (foldr, isPrefixOf)
import Control.Monad
import Prelude hiding (writeFile)

-- 
-- GenerateArbitraryTests.hs
-- A set of functions for generating a set of programs and storing their JSON encodings
-- in a file.
-- 

-- Typical term lengths: Easy = 10, Medium = 15, Hard = 20, VeryHard = 25

data Config = Config {forFileName :: String,
                      forCount :: Int,
                      difficulty :: Difficulty,
                      termLength :: Int}

defaultConfig :: Config
defaultConfig = Config {forFileName = "arbitraryForList.json", forCount = 50000, difficulty = Easy, termLength = 10}

parseArgumentsHelper :: Config -> String -> IO Config
parseArgumentsHelper cfg opt | "-forFileName=" `isPrefixOf` opt = pure $ cfg {forFileName = drop 13 opt}
                             | "-forCount=" `isPrefixOf` opt = pure $ cfg {forCount = read $ drop 10 opt}
                             | "-difficulty=" `isPrefixOf` opt = pure $ cfg {difficulty = read $ drop 12 opt}
                             | "-termLength=" `isPrefixOf` opt = pure $ cfg {termLength = read $ drop 12 opt}
                             | otherwise = die "You used an option that wasn't present."

parseArguments :: IO Config
parseArguments = do args <- getArgs
                    foldM parseArgumentsHelper defaultConfig args

generateArbitraryFor :: Difficulty -> Int -> Int -> IO [ProgFor]
generateArbitraryFor difficulty count exprLength = generate $ vectorOf count $ fst <$> (arbitrarySizedProgForWithDifficulty difficulty 0 exprLength)


main :: IO ()
main = do cfg <- parseArguments
          for_progs <- generateArbitraryFor (difficulty cfg) (forCount cfg) (termLength cfg)
          writeFile (show (difficulty cfg) ++ "-" ++ (forFileName cfg)) $ encode for_progs

