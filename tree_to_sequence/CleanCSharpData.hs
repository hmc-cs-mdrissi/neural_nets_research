{-# LANGUAGE OverloadedStrings #-}
module CleanCSharpData where

import qualified Data.Text as T
import Control.Arrow (first)

import System.Environment (getArgs)
import System.IO (readFile)

remove_third_real_quote :: T.Text -> T.Text
remove_third_real_quote str = uncurry T.append $ first T.init $ T.splitAt (find_third_real_quote_helper str 0 0) str

find_third_real_quote_helper :: T.Text -> Int -> Int -> Int
find_third_real_quote_helper str n pos | T.head str == '"' = if n == 2 then pos else find_third_real_quote_helper (T.tail str) (n + 1) (pos + 1)
                                       | T.head str == '\\' = find_third_real_quote_helper (T.drop 2 str) n (pos + 2)
                                       | otherwise = find_third_real_quote_helper (T.tail str) n (pos + 1)

remove_double_quotes :: T.Text -> T.Text -> T.Text
remove_double_quotes prefix str = case T.splitOn prefix str of
                                    h : t -> T.intercalate prefix $ h : map (remove_third_real_quote . T.tail) t
                                    [] -> error "Not a possible situation."

main :: IO ()
main = do [filename] <- getArgs
          programs <- readFile filename
          putStrLn $ T.unpack $ remove_double_quotes "\"StringLiteralExpression\":" (T.pack programs)
