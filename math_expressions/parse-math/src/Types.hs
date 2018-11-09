module Types where
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
  deriving (Show, Eq)

data ContainerSymbol = AbsBar | LeftParen | RightParen | LeftBrace | RightBrace| Magnitude deriving (Show, Eq)

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
    Neq deriving (Show, Eq)

data UnOp = 
    Sin | Cos | Tan | Sqrt | NegSign deriving (Show, Eq) 
    
data DoubOp =
    FracOp | LogOp | LimOp deriving (Show, Eq) 

data Symbol = 
    Alpha |
    Beta |
    Gamma | 
    Phi |
    Pi |
    Theta |
    Infty | 
    Ldots deriving (Show, Eq) 

data PUnOp = Factorial deriving (Show, Eq) 

data BarOp = Bar deriving (Show, Eq)

-- instance Show MathExpression where
--     show = showExpr 0
