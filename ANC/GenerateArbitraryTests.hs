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

data Config = Config {forFileName :: String,
                      forCount :: Int}

defaultConfig :: Config
defaultConfig = Config {forFileName = "arbitraryForList.json", forCount = 50000}

parseArgumentsHelper :: Config -> String -> IO Config
parseArgumentsHelper cfg opt | "-forFileName=" `isPrefixOf` opt = pure $ cfg {forFileName = drop 13 opt}
                             | "-forCount=" `isPrefixOf` opt = pure $ cfg {forCount = read $ drop 10 opt}
                             | otherwise = die "You used an option that wasn't present."

parseArguments :: IO Config
parseArguments = do args <- getArgs
                    foldM parseArgumentsHelper defaultConfig args

generateArbitraryFor :: Int -> Int -> IO [ProgFor]
generateArbitraryFor count exprLength = generate $ vectorOf count $ fst <$> (arbitrarySizedProgFor [2, 1, 1, 1] 0 exprLength)


main :: IO ()
main = do cfg <- parseArguments
          for_progs <- generateArbitraryFor (forCount cfg) 30
          writeFile (forFileName cfg) $ encode for_progs

