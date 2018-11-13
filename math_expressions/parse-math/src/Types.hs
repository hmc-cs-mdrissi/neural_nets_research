{-# LANGUAGE DeriveGeneric #-}

module Types where
  -- Need to export all constructors...
  -- ( MathExpression
  -- , ContainerSymbol
  -- , BinOp
  -- , UnOp
  -- , DoubOp
  -- , Symbol
  -- , PUnOp
  -- ) where

import Data.Aeson
import GHC.Generics

data MathExpression
  = IntegerM Integer
  -- | DoubleM Double
  | VarName Char
  | Nil
  | Symbol Symbol
  | Container ContainerSymbol MathExpression ContainerSymbol
  | UnOp UnOp MathExpression
  | UnOpExp UnOpExp MathExpression MathExpression
  | PUnOp MathExpression PUnOp
  | DoubOp DoubOp MathExpression MathExpression
  | BinOp MathExpression BinOp MathExpression
  | Digit MathExpression Integer
  | Sum MathExpression MathExpression MathExpression
  | Integral MathExpression MathExpression MathExpression -- think about this more.  do we want bounds?  integraion vars?
  deriving (Show, Eq, Generic)

data ContainerSymbol = AbsBar | LeftParen | RightParen | LeftBrace | RightBrace| Magnitude deriving (Show, Eq, Generic)

data BinOp =
    Plus |
    Minus |
    Div |
    Mult |
    BinaryPm |
    Equal |
    Marrow |
    SubscriptOp |
    SuperscriptOp |
    ImplicitMult |
    Le |
    Leq |
    Ge |
    Geq |
    Neq deriving (Show, Eq, Generic)

data UnOpExp = Sin | Cos | Tan deriving (Show, Eq, Generic)

data UnOp = Sqrt | NegSign | UnaryPm deriving (Show, Eq, Generic)

data DoubOp =
    FracOp | LogOp | LimOp deriving (Show, Eq, Generic)

data Symbol =
    Alpha |
    Beta |
    Gamma |
    Phi |
    Pi |
    Theta |
    Infty |
    Cdots |
    Ldots deriving (Show, Eq, Generic)

data PUnOp = Factorial | Whatever deriving (Show, Eq, Generic)

instance ToJSON MathExpression
instance ToJSON ContainerSymbol
instance ToJSON BinOp
instance ToJSON UnOp
instance ToJSON UnOpExp
instance ToJSON DoubOp
instance ToJSON Symbol
instance ToJSON PUnOp
