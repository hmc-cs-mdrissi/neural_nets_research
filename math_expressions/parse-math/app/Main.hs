module Main where

import Types
import Parser

import Data.Aeson
import qualified Data.ByteString.Lazy as B
import Data.Map (Map)
import qualified Data.Map.Strict as Map
import Debug.Trace (trace)


main :: IO()
main = do
  -- file <- trace "doing things" (B.readFile "../single_expr.json")
  -- file <- B.readFile "../TEST2016_INKML_GT_GET_Strings.json"
  -- file <- B.readFile "../2012_dataset.json"
  file <- B.readFile "../2011_dataset_test.json"
  case decode file :: Maybe (Map String String) of
    Just loadedExprs -> 
      either print (B.writeFile "../2011_parsed_test_dset.json" . encode ) $
        mapM parseMath (Map.elems loadedExprs)
    Nothing -> print "Unparsable"


-- main