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

kill_single_quotes :: T.Text -> T.Text -> T.Text
kill_single_quotes prefix str = case T.splitOn prefix str of
                                    h : t -> T.intercalate (T.init prefix) $ h : map (remove_second_char) t
                                    [] -> error "Not a possible situation."


remove_second_char :: T.Text -> T.Text 
remove_second_char str = if T.head str == '"' 
                         then T.concat [T.singleton '\\', T.take 1 str, T.drop 2 str] -- if the string starts with ", escape it
                         else 
                            if T.take 2 str == T.pack "\\'"
                            then T.drop 2 str-- if the string starts with \', remove the escape
                            else
                                if T.head str == '\\' 
                                then T.concat [T.take 2 str, T.drop 3 str] -- keep all other escapes, which means we kill the third character
                                else T.concat [T.take 1 str, T.drop 2 str] -- just chop out second char (') as usual

main :: IO ()
main = do [filename] <- getArgs
          programs <- readFile filename
          let slightlyLessBrokenPrograms = T.unpack $ remove_double_quotes "\"StringLiteralExpression\":" (T.pack programs)
          putStrLn $ T.unpack $ kill_single_quotes "\"CharacterLiteralExpression\":\"'" (T.pack slightlyLessBrokenPrograms)

