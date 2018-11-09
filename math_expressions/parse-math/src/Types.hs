{-# LANGUAGE DeriveGeneric #-}

module Syntax where
  -- Need to export all constructors...
  -- ( MathExpression
  -- , ContainerSymbol
  -- , NatBinOp
  -- , UnOp
  -- , DoubOp
  -- , Symbol
  -- , PUnOp
  -- , BarOp
  -- ) where

import Data.Aeson
import GHC.Generics

data MathExpression
  = IntegerM Integer
  | DoubleM Double
  | VarName Char
  | Nil
  | Symbol Symbol
  | Container ContainerSymbol MathExpression ContainerSymbol
  | UnOp UnOp MathExpression
  | PUnOp MathExpression PUnOp
  | DoubOp DoubOp MathExpression MathExpression
  | NatBinOp MathExpression NatBinOp MathExpression
  | Sum MathExpression MathExpression MathExpression
  deriving (Show, Eq, Generic)

data ContainerSymbol = AbsBar | LeftParen | RightParen | LeftBrace | RightBrace| Magnitude deriving (Show, Eq, Generic)

data NatBinOp =
    Plus |
    Minus |
    Div |
    Mult |
    PlusMinus |
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

data UnOp =
    Sin | Cos | Tan | Sqrt | NegSign deriving (Show, Eq, Generic)

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
    Ldots deriving (Show, Eq, Generic)

data PUnOp = Factorial deriving (Show, Eq, Generic)

data BarOp = Bar deriving (Show, Eq, Generic)

instance ToJSON MathExpression
instance ToJSON ContainerSymbol
instance ToJSON NatBinOp
instance ToJSON UnOp
instance ToJSON DoubOp
instance ToJSON Symbol
instance ToJSON PUnOp
