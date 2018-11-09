module Main where

import Types
import Parser

import Data.Aeson
import qualified Data.ByteString.Lazy as B
import Data.Map (Map)
import qualified Data.Map.Lazy as Map

main :: IO()
main = do
  a <- B.readFile "single_expr.json"
  -- a <- B.readFile "TEST2016_INKML_GT_GET_Strings.json"
  case decode a :: Maybe (Map String String) of
    Just loadedExprs -> B.writeFile "just_testing.json" $ encode (Map.map (parseMath) (loadedExprs))
    Nothing -> print "Unparsable"
  putStrLn "hi"

-- main = putStrLn parseString
