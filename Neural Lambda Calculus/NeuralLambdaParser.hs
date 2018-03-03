module NeuralLambdaParser where

import Text.Parsec.Language (emptyDef)
import qualified Text.Parsec.Token as P
import Text.Parsec.Char
import Text.Parsec.Prim
import Text.Parsec.Combinator
import Text.Parsec.String (Parser)
import Control.Applicative (some, many)
import Data.String
import Data.Function
import Data.Bifunctor
import Text.Parsec.Error

--Let expressions are not part of the definition because let expressions can be viewed as syntactic sugar.
--let x = y in z is the same as (lambda x.z) y.
data NeuralLambdaExpression = 
    Variable String | 
    Abstraction String NeuralLambdaExpression | 
    Application NeuralLambdaExpression NeuralLambdaExpression | 
    Natural Integer | 
    Nil | 
    Cons NeuralLambdaExpression NeuralLambdaExpression | 
    Foldr NeuralLambdaExpression NeuralLambdaExpression NeuralLambdaExpression | 
    NatBinOp NatBinOp NeuralLambdaExpression NeuralLambdaExpression |
    NeuralVariable [(String,Float)] |
    Prob ([(String,Float)],Float) ((String, NeuralLambdaExpression), Float) ((NeuralLambdaExpression, NeuralLambdaExpression),Float) |
    Combo [(NeuralLambdaExpression,Float)] deriving Eq

data NatBinOp =
    Plus | Minus | Div | Mult deriving Eq

instance Show NeuralLambdaExpression where
    show = showExpr 0

showExpr :: Int -> NeuralLambdaExpression -> String
showExpr 0 (Abstraction var body) = "lambda " ++  var ++ ". " ++ showExpr 0 body 
showExpr 0 a = showExpr 1 a

showExpr 1 (NatBinOp Plus x y) = showExpr 1 x ++ " + " ++ showExpr 2 y
showExpr 1 (NatBinOp Minus x y) = showExpr 1 x ++ " - " ++ showExpr 2 y
showExpr 1 a = showExpr 2 a

showExpr 2 (NatBinOp Mult x y) = showExpr 2 x ++ " * " ++ showExpr 3 y
showExpr 2 (NatBinOp Div x y) = showExpr 2 x ++ " / " ++ showExpr 3 y
showExpr 2 a = showExpr 3 a

showExpr 3 (Application x y) = showExpr 3 x ++ " " ++ showExpr 4 y
showExpr 3 (Foldr x y z) = "foldrlc " ++ showExpr 4 x ++ " " ++ showExpr 4 y ++ " " ++ showExpr 4 z
showExpr 3 (Cons x y) = "cons " ++ showExpr 4 x ++ " " ++ showExpr 4 y 
showExpr 3 a = showExpr 4 a

showExpr 4 (Natural n) = show n
showExpr 4 (Variable a) = a
showExpr 4 Nil = "[]"
showExpr 4 (NeuralVariable vars) = "{" ++ show vars ++ "}"
showExpr 4 (Prob (vars, p1) ((var,body), p2) ((nlc1,nlc2), p3)) = "{" ++ show vars ++ " : " ++ show p1 ++ " , " 
  ++ show (Abstraction var body) ++ " : " ++ show p2 ++ " , " ++ show (Application nlc1 nlc2) ++ " : " ++ show p3
showExpr 4 (Combo lst) = "{" ++ show lst ++ "}"
showExpr _ a = "(" ++ showExpr 0 a ++ ")"

substitute :: NeuralLambdaExpression -> String -> NeuralLambdaExpression -> NeuralLambdaExpression
substitute (Variable x) var expr | x == var = expr
                                  | otherwise = Variable x
substitute x@(Abstraction bound expr1) var expr2 | bound == var =  x
                                                  | otherwise = Abstraction bound $ substitute expr1 var expr2
substitute (Application expr1 expr2) var expr3 = Application (substitute expr1 var expr3) (substitute expr2 var expr3)
substitute n@(Natural _) _ _ = n
substitute Nil _ _ = Nil
substitute (Cons expr1 expr2) var expr = Cons (substitute expr1 var expr) (substitute expr2 var expr)
substitute (Foldr expr1 expr2 expr3) var expr = Foldr (substitute expr1 var expr) (substitute expr2 var expr) (substitute expr3 var expr)
substitute (NatBinOp op expr1 expr2) var expr = NatBinOp op (substitute expr1 var expr) (substitute expr2 var expr)

neuralLambdaDef :: P.LanguageDef ()
neuralLambdaDef = emptyDef {P.identStart = letter
                     ,P.identLetter = alphaNum <|> char '\''
                     ,P.opStart = oneOf "=.[];"
                     ,P.opLetter = oneOf "=.[];+-/*"
                     ,P.reservedOpNames = ["=", ".", "[", "]", ";","+","-","/","*"]
                     ,P.reservedNames = ["lambda", "let", "in", "foldrlc","cons"]
                     }

lexer :: P.TokenParser ()
lexer = P.makeTokenParser neuralLambdaDef

reserved :: String -> Parser ()
reserved = P.reserved lexer

equal :: Parser ()
equal = P.reservedOp lexer "="

dot :: Parser ()
dot = P.reservedOp lexer "."

listOpen :: Parser ()
listOpen = char '[' *> whiteSpace

listClose :: Parser ()
listClose = char ']' *> whiteSpace

semicolon :: Parser ()
semicolon = P.reservedOp lexer ";"

plus :: Parser ()
plus = P.reservedOp lexer "+"

minus :: Parser ()
minus = P.reservedOp lexer "-"

divide :: Parser ()
divide = P.reservedOp lexer "/"

mult :: Parser ()
mult = P.reservedOp lexer "*"

identifier :: Parser String
identifier = P.identifier lexer

parens :: Parser NeuralLambdaExpression -> Parser NeuralLambdaExpression
parens = P.parens lexer

lexeme :: Parser a -> Parser a
lexeme = P.lexeme lexer

whiteSpace :: Parser ()
whiteSpace = P.whiteSpace lexer

variable :: Parser NeuralLambdaExpression
variable = Variable <$> identifier

abstraction :: Parser NeuralLambdaExpression
abstraction = do reserved "lambda"
                 x <- identifier
                 dot
                 y <- neuralLambdaExpression' 0 <?> "lambda expression"
                 return $ Abstraction x y

                          
letExpression :: Parser NeuralLambdaExpression
letExpression =  do reserved "let"
                    x <- identifier
                    equal
                    y <- neuralLambdaExpression' 0 <?> "lambda expression"
                    reserved "in"
                    z <- neuralLambdaExpression' 0 <?> "lambda expression"
                    return $ substitute z x y

natural :: Parser NeuralLambdaExpression
natural = Natural <$> P.natural lexer

nil :: Parser NeuralLambdaExpression
nil = do listOpen
         listClose
         return $ Nil

cons :: Parser NeuralLambdaExpression
cons = do reserved "cons"
          n <- neuralLambdaExpression' 3 <?> "lambda expression"
          l <- neuralLambdaExpression' 3 <?> "lambda expression"
          return $ Cons n l

listHelp :: Parser NeuralLambdaExpression
listHelp = neuralLambdaExpression' 0 `chainr1` (semicolon *> return Cons)

placeNil :: NeuralLambdaExpression -> NeuralLambdaExpression
placeNil (Cons x y@(Cons _ _)) = Cons x (placeNil y)
placeNil (Cons x y) = Cons x (Cons y Nil)
placeNil x = Cons x Nil

list :: Parser NeuralLambdaExpression
list = try nil <|> placeNil <$> (listOpen *> listHelp <* listClose)

foldrlc :: Parser NeuralLambdaExpression
foldrlc = do reserved "foldrlc"
             x <- neuralLambdaExpression' 3 <?> "lambda expression"
             y <- neuralLambdaExpression' 3 <?> "lambda expression"
             z <- neuralLambdaExpression' 3 <?> "lambda expression"
             return $ Foldr x y z

neuralLambdaExpression' :: Int -> Parser NeuralLambdaExpression
neuralLambdaExpression' 0 = neuralLambdaExpression' 1 `chainl1` ((plus *> pure (NatBinOp Plus)) <|> (minus *> pure (NatBinOp Minus)))
neuralLambdaExpression' 1 = neuralLambdaExpression' 2 `chainl1` ((mult *> pure (NatBinOp Mult)) <|> (divide *> pure (NatBinOp Div)))
neuralLambdaExpression' 2 = neuralLambdaExpression' 3 `chainl1` (lexeme $ return Application)
neuralLambdaExpression' 3 = letExpression <|> abstraction <|> foldrlc <|> variable <|> natural <|> cons <|> list <|> parens (neuralLambdaExpression' 0)

parseLambda :: String -> Either ParseError NeuralLambdaExpression
parseLambda = parse (whiteSpace *> neuralLambdaExpression' 0) ""