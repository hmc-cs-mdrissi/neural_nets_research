module LambdaParser where

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
data LambdaExpression = 
    Variable String | 
    Abstraction String LambdaExpression | 
    Application LambdaExpression LambdaExpression | 
    Natural Integer | 
    Nil | 
    Cons LambdaExpression LambdaExpression | 
    Foldr LambdaExpression LambdaExpression LambdaExpression | 
    NatBinOp NatBinOp LambdaExpression LambdaExpression deriving Eq

data NatBinOp =
    Plus | Minus | Div | Mult deriving Eq

instance Show LambdaExpression where
    show = showExpr 0

showExpr :: Int -> LambdaExpression -> String
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
showExpr _ a = "(" ++ showExpr 0 a ++ ")"

substitute :: LambdaExpression -> String -> LambdaExpression -> LambdaExpression
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

lambdaDef :: P.LanguageDef ()
lambdaDef = emptyDef {P.identStart = letter
                     ,P.identLetter = alphaNum <|> char '\''
                     ,P.opStart = oneOf "=.[];"
                     ,P.opLetter = oneOf "=.[];+-/*"
                     ,P.reservedOpNames = ["=", ".", "[", "]", ";","+","-","/","*"]
                     ,P.reservedNames = ["lambda", "let", "in", "foldrlc","cons"]
                     }

lexer :: P.TokenParser ()
lexer = P.makeTokenParser lambdaDef

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

parens :: Parser LambdaExpression -> Parser LambdaExpression
parens = P.parens lexer

lexeme :: Parser a -> Parser a
lexeme = P.lexeme lexer

whiteSpace :: Parser ()
whiteSpace = P.whiteSpace lexer

variable :: Parser LambdaExpression
variable = Variable <$> identifier

abstraction :: Parser LambdaExpression
abstraction = do reserved "lambda"
                 x <- identifier
                 dot
                 y <- lambdaExpression' 0 <?> "lambda expression"
                 return $ Abstraction x y

                          
letExpression :: Parser LambdaExpression
letExpression =  do reserved "let"
                    x <- identifier
                    equal
                    y <- lambdaExpression' 0 <?> "lambda expression"
                    reserved "in"
                    z <- lambdaExpression' 0 <?> "lambda expression"
                    return $ substitute z x y

natural :: Parser LambdaExpression
natural = Natural <$> P.natural lexer

nil :: Parser LambdaExpression
nil = do listOpen
         listClose
         return $ Nil

cons :: Parser LambdaExpression
cons = do reserved "cons"
          n <- lambdaExpression' 3 <?> "lambda expression"
          l <- lambdaExpression' 3 <?> "lambda expression"
          return $ Cons n l

listHelp :: Parser LambdaExpression
listHelp = lambdaExpression' 0 `chainr1` (semicolon *> return Cons)

placeNil :: LambdaExpression -> LambdaExpression
placeNil (Cons x y@(Cons _ _)) = Cons x (placeNil y)
placeNil (Cons x y) = Cons x (Cons y Nil)
placeNil x = Cons x Nil

list :: Parser LambdaExpression
list = try nil <|> placeNil <$> (listOpen *> listHelp <* listClose)

foldrlc :: Parser LambdaExpression
foldrlc = do reserved "foldrlc"
             x <- lambdaExpression' 3 <?> "lambda expression"
             y <- lambdaExpression' 3 <?> "lambda expression"
             z <- lambdaExpression' 3 <?> "lambda expression"
             return $ Foldr x y z

lambdaExpression' :: Int -> Parser LambdaExpression
lambdaExpression' 0 = lambdaExpression' 1 `chainl1` ((plus *> pure (NatBinOp Plus)) <|> (minus *> pure (NatBinOp Minus)))
lambdaExpression' 1 = lambdaExpression' 2 `chainl1` ((mult *> pure (NatBinOp Mult)) <|> (divide *> pure (NatBinOp Div)))
lambdaExpression' 2 = lambdaExpression' 3 `chainl1` (lexeme $ return Application)
lambdaExpression' 3 = letExpression <|> abstraction <|> foldrlc <|> variable <|> natural <|> cons <|> list <|> parens (lambdaExpression' 0)

parseLambda :: String -> Either ParseError LambdaExpression
parseLambda = parse (whiteSpace *> lambdaExpression' 0) ""