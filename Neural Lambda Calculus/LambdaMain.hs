module LambdaMain where

import System.Environment
import System.Exit
import System.IO

import Data.Maybe
import Data.List
import qualified Data.Set as Set

import LambdaInterp
import LambdaParser

import Text.Parsec.Error

import Control.Monad

data Config = Config {contents :: String,
                      checkMode :: Bool, 
                      churchMode :: Bool}


defaultConfig :: Config
defaultConfig = Config {contents = "", checkMode = False, churchMode = False}

convertArgs :: [String] -> [String]
convertArgs arguments = do argument <- arguments
                           if "-" `isPrefixOf` argument
                                then map (\opt -> '-':[opt]) (tail argument)
                                else pure argument

constructConfigChruch :: Config -> IO (Config, [String])
constructConfigChruch start = do args <- getArgs
                                 let arguments = convertArgs args
                                 if "-n" `elem` arguments 
                                    then return (start {churchMode = True}, delete "-n" arguments) 
                                    else return (start, arguments)

constructConfigCheck :: (Config, [String]) -> (Config, [String])
constructConfigCheck (start, arguments) =   if "-c" `elem` arguments 
                                                then (start {checkMode = True}, delete "-c" arguments) 
                                                else (start, arguments)

constructConfigContents :: (Config, [String]) -> IO Config
constructConfigContents (start, arguments) = case listToMaybe arguments of
                                                Nothing -> getContents >>= \str -> return (start {contents = str})
                                                Just "-" -> getContents >>= \str -> return (start {contents = str})
                                                Just filename -> readFile filename >>= \str -> return (start {contents = str})

constructConfig :: IO Config
constructConfig = constructConfigCheck <$> constructConfigChruch defaultConfig >>= constructConfigContents

doCheck :: LambdaExpression -> IO ()
doCheck expr = let unbound_variables = free_variables $ convert_to_one_argument expr 
               in unless (Set.null unbound_variables) (die $ "Unbound Variables: " ++ intercalate ", " (Set.toList unbound_variables))

interpretExpr :: LambdaExpression -> Config -> IO ()
interpretExpr expr options = do when (checkMode options) (doCheck expr)
                                case evalCBV' expr of 
                                    Left err -> die err
                                    Right expr' -> useEvaluatedExpr expr' options

useEvaluatedExpr :: LambdaExpression -> Config -> IO ()
useEvaluatedExpr expr options = if churchMode options 
                                    then case convert_to_numeral' expr of
                                            Left err -> die err
                                            Right num -> print num
                                    else print expr

main :: IO ()
main = do config <- constructConfig
          case parseLambda (contents config) of
            Left err -> die $ show err
            Right expr -> interpretExpr expr config