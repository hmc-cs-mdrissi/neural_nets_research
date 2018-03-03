{-# LANGUAGE DeriveGeneric #-}

module LambdaParser where

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


data ExprLambda = VarL String | ConstL Integer | PlusL ExprLambda ExprLambda | MinusL ExprLambda ExprLambda deriving (Show, Generic)
data CmpLambda = EqualL ExprLambda ExprLambda | LeL ExprLambda ExprLambda | GeL ExprLambda ExprLambda deriving (Show, Generic)
data ProgLambda = UnitLambda | If CmpLambda ProgLambda ProgLambda | ExprL ExprLambda
                | LetLambda String ProgLambda ProgLambda | LetRecLambda String String ProgLambda ProgLambda deriving (Show, Generic)

instance ToJSON ExprLambda where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON ExprLambda

instance ToJSON CmpLambda where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON CmpLambda

instance ToJSON ProgLambda where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON ProgLambda

progLambdaDef :: P.LanguageDef ()
progLambdaDef = emptyDef {P.identStart = letter,
                          P.identLetter = alphaNum,
                          P.opStart = P.opLetter progLambdaDef,
                          P.opLetter = oneOf "+-=;><",
                          P.reservedOpNames = [">", "<", "==", "=", "+", "-"],
                          P.reservedNames = ["if", "then", "else", "let", "in", "letrec", "unit"]}

lexer :: P.TokenParser ()
lexer = P.makeTokenParser progLambdaDef

reserved :: String -> Parser ()
reserved = P.reserved lexer

reservedOp :: String -> Parser ()
reservedOp = P.reserved lexer

double_equal :: Parser ()
double_equal = P.reservedOp lexer "=="

equal :: Parser ()
equal = P.reservedOp lexer "="

plus :: Parser ()
plus = P.reservedOp lexer "+"

minus :: Parser ()
minus = P.reservedOp lexer "-"

le :: Parser ()
le = P.reservedOp lexer "<"

ge :: Parser ()
ge = P.reservedOp lexer ">"

identifier :: Parser String
identifier = P.identifier lexer

parens :: Parser a -> Parser a
parens = P.parens lexer

exprL, exprL_term, varL, constL :: Parser ExprLambda
exprL = exprL_term `chainl1` ((plus *> pure PlusL) <|> (minus *> pure MinusL))
exprL_term = varL <|> constL
varL = VarL <$> identifier
constL = ConstL <$> P.integer lexer

cmpP :: Parser CmpLambda
cmpP = try (liftA2 EqualL (exprL <* double_equal) exprL) <|>
       try (liftA2 LeL (exprL <* le) exprL) <|>
       liftA2 (GeL) (exprL <* ge) exprL

progP, unitP, ifP, letP, letrecP :: Parser ProgLambda
progP = unitP <|> ifP <|> letP <|> letrecP <|> ExprL <$> exprL <|> parens progP

unitP = reserved "unit" *> pure UnitLambda

ifP = do
        reserved "if"
        cond <- cmpP
        reserved "then"
        if_body <- progP
        reserved "else"
        else_body <- progP
        return $ If cond if_body else_body

letP = do
        reserved "let"
        var <- identifier
        equal
        value <- progP
        reserved "in"
        body <- progP
        return $ LetLambda var value body


letrecP = do
            reserved "letrec"
            var1 <- identifier
            var2 <- identifier
            equal
            value <- progP
            reserved "in"
            body <- progP
            return $ LetRecLambda var1 var2 value body

parseProg :: String -> Either ParseError ProgLambda
parseProg = parse progP ""
