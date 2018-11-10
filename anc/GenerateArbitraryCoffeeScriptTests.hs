module GenerateArbitraryCoffeeScriptTests where

import Test.QuickCheck
import Data.Aeson

import ArbitraryCoffeeScriptTests

import System.Environment
import System.Exit
import qualified Data.ByteString.Lazy as B
import Data.List (isPrefixOf)
import Control.Monad

--
-- GenerateArbitraryCoffeeScriptTests.hs
-- A set of functions for generating a set of coffeescript programs and storing both their json encodings and all of the programs.
--

data Config = Config {csFileName :: String,
                      csCount :: Int,
                      csFolder :: String,
                      termLength :: Int}

-- 28 is a term length that gives programs of roughly similar size as the programs used in the tree to tree program translation paper.
-- 32 is a small increase from there to try to make the dataset a bit harder to make differentiating models easier.

defaultConfig :: Config
defaultConfig = Config {csFileName="test_cs.json", csCount=10, csFolder="cs_programs/", termLength=32}

parseArgumentsHelper :: Config -> String -> IO Config
parseArgumentsHelper cfg opt | "-csFileName=" `isPrefixOf` opt = pure $ cfg {csFileName = drop 12 opt}
                             | "-csCount=" `isPrefixOf` opt = pure $ cfg {csCount = read $ drop 9 opt}
                             | "-csFolder=" `isPrefixOf` opt = pure $ cfg {csFolder = drop 10 opt}
                             | "-termLength=" `isPrefixOf` opt = pure $ cfg {termLength = read $ drop 12 opt}
                             | otherwise = die "You used an option that wasn't present."

parseArguments :: IO Config
parseArguments = do args <- getArgs
                    foldM parseArgumentsHelper defaultConfig args

generateArbitraryCS :: Int -> Int -> IO [CoffeeScript]
generateArbitraryCS count termLength = generate $ vectorOf count $ fst <$> (arbitrarySizedCoffeeScript (-1) termLength)

constructFilesCS :: String -> [CoffeeScript] -> IO ()
constructFilesCS folder_name progs = mapM_ (\(i, prog) -> writeFile (folder_name ++ show i ++ ".coffee") (show prog)) (zip [1..] progs)

main :: IO ()
main = do cfg <- parseArguments
          cs_progs <- generateArbitraryCS (csCount cfg) (termLength cfg)
          constructFilesCS (csFolder cfg) cs_progs
          B.writeFile (csFileName cfg) $ encode cs_progs
