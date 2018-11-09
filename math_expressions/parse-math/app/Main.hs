module Main where

import Types
import Parser

import Data.Aeson
import qualified Data.ByteString.Lazy as B
import Data.Map (Map)
import qualified Data.Map.Strict as Map

main :: IO()
main = do
  file <- B.readFile "../single_expr.json"
  -- a <- B.readFile "TEST2016_INKML_GT_GET_Strings.json"
  case decode file :: Maybe (Map String String) of
    Just loadedExprs ->
      either print (B.writeFile "just_testing.json" . encode ) $
        mapM parseMath (Map.elems loadedExprs)
    Nothing -> print "Unparsable"
  putStrLn "hi"

-- main = putStrLn parseString
