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

data Expr = Var String | Const Integer | Plus Expr Expr | Minus Expr Expr deriving (Show, Generic)
data Cmp = Equal Expr Expr | Le Expr Expr | Ge Expr Expr deriving (Show, Generic)

instance ToJSON Expr where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON Expr

instance ToJSON Cmp where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON Cmp