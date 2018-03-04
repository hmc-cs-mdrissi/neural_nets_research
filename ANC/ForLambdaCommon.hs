{-# LANGUAGE DeriveGeneric #-}

module ForLambdaCommon where

import Text.Parsec.Language (emptyDef)
import qualified Text.Parsec.Token as P
import Text.Parsec.Char
import Text.Parsec.Prim
import Text.Parsec.Combinator
import Text.Parsec.String (Parser)
import Text.Parsec.Error
import Control.Applicative (liftA2)
import Data.Aeson
import GHC.Generics

data Expr = Var String | Const Integer | Plus Expr Expr | Minus Expr Expr deriving Generic
data Cmp = Equal Expr Expr | Le Expr Expr | Ge Expr Expr deriving Generic

showExpr :: Int -> Expr -> String
showExpr 0 (Plus x y) = (showExpr 0 x) ++ " + " ++ (showExpr 1 y)
showExpr 0 (Minus x y) = (showExpr 0 x) ++ " + " ++ (showExpr 1 y)
showExpr 0 a = showExpr 1 a

showExpr 1 (Var x) = x
showExpr 1 (Const n) = show n
showExpr 1 a = "(" ++ (showExpr 0 a) ++ ")"

instance Show Expr where
    show = showExpr 0

instance Show Cmp where
    show (Equal x y) = show x ++ " == " ++ show y
    show (Le x y) = show x ++ " < " ++ show y
    show (Ge x y) = show x ++ " > " ++ show y  

instance ToJSON Expr where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON Expr

instance ToJSON Cmp where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON Cmp