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
                      lambdaFileName :: String, 
                      forCount :: Int,
                      lambdaCount :: Int}


defaultConfig :: Config
defaultConfig = Config {forFileName = "arbitraryForList.json", lambdaFileName = "arbitraryLambdaList.json", forCount = 50000, lambdaCount = 50000}

parseArgumentsHelper :: Config -> String -> IO Config
parseArgumentsHelper cfg opt | "-forFileName=" `isPrefixOf` opt = pure $ cfg {forFileName = drop 13 opt}
                             | "-lambdaFileName=" `isPrefixOf` opt = pure $ cfg {lambdaFileName = drop 16 opt}
                             | "-forCount=" `isPrefixOf` opt = pure $ cfg {forCount = read $ drop 10 opt}
                             | "-lambdaCount=" `isPrefixOf` opt = pure $ cfg {lambdaCount = read $ drop 13 opt}
                             | otherwise = die "You used an option that wasn't present."

parseArguments :: IO Config
parseArguments = do args <- getArgs
                    foldM parseArgumentsHelper defaultConfig args

generateArbitraryFor :: Int -> Int -> IO [ProgFor]
generateArbitraryFor count exprLength = generate $ vectorOf count $ arbitrarySizedProgFor exprLength

generateArbitraryProgLambda :: Int -> Int -> IO [ProgLambda]
generateArbitraryProgLambda count exprLength = generate $ vectorOf count $ arbitrarySizedProgLambda exprLength

main :: IO ()
main = do cfg <- parseArguments
          for_progs <- generateArbitraryFor (forCount cfg) 30
          lambda_progs <- generateArbitraryProgLambda (lambdaCount cfg) 30
          writeFile (forFileName cfg) $ encode for_progs
          writeFile (lambdaFileName cfg) $ encode lambda_progs